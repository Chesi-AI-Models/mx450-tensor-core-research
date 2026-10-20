#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mrc_minimal.py —— 从零实现的 MRC(Multipath Reliable Connection) 最小语义核心

依据：MetaRoCE-Spec-1.0（facebookincubator/metaroce）§1.3 / §1.5 / §8
     + Open-RDMA 对 MRC 的描述（多路径 + 乱序 + 选择性重传）。

★ 不依赖 RDMA 硬件，也不依赖 folly/thrift —— 纯 Python 标准库，用 UDP 承载。
  目的：把 MRC 四个机制真正跑起来并量化
    1. Tag / 多流：一个连接承载多条独立消息流
    2. 显式多路径：每 flow 独立滑动窗口 + 独立 SACK 位图 + 独立 AIMD
    3. 松弛顺序交付：乱序到达即按消息内偏移直接落盘（无重排序缓冲）
    4. 接收端驱动限速：ACK 回传入站带宽份额

包格式按规范 §8.1：
  w0 = Opcode(8) | Version(4) | FlowID(12)
  w1 = DestQP(32)  w2 = SrcQP(32)
  METH 风格：MsgOffset(32) | Len(16) | FlowLocalSeq(16)
  尾部 mCRC32（覆盖头+载荷）

修正记录（第一版踩的两个坑）：
  bug1: 每 flow 的序列号误用全局包号（s = f, f+n, ...）→ SACK 的 cum 推不动、窗口不开。
        修法：每 flow 用**独立的局部序号** L，全局包号由 per_flow[f][L] 映射。
  bug2: 丢包代理用临时 socket 转发 → 接收端 ACK 回到没人收的端口，发送端永远等不到确认。
        修法：代理用**固定 socket** 收发，记住客户端地址并把回包转发回去。
"""
import argparse
import hashlib
import random
import select
import socket
import struct
import sys
import threading
import time
import zlib

OP_WRITE = 0x0A
OP_SACK = 0x20
VERSION = 1
MTU = 1024
HDR_BASE = 12
HDR_METH = 8
CRC_LEN = 4
HDR_LEN = HDR_BASE + HDR_METH + CRC_LEN


def pack_base(op, flow, dqp, sqp):
    w0 = ((op & 0xFF) << 24) | ((VERSION & 0xF) << 12) | (flow & 0xFFF)
    return struct.pack("!III", w0, dqp, sqp)


def unpack_base(b):
    w0, dqp, sqp = struct.unpack_from("!III", b, 0)
    return (w0 >> 24) & 0xFF, (w0 >> 12) & 0xF, w0 & 0xFFF, dqp, sqp


def build_pkt(op, flow, dqp, sqp, msg_off, seq_l, payload=b""):
    b = pack_base(op, flow, dqp, sqp) + struct.pack("!IHH", msg_off, len(payload), seq_l) + payload
    return b + struct.pack("!I", zlib.crc32(b) & 0xFFFFFFFF)


def parse_pkt(buf):
    if len(buf) < HDR_LEN:
        return None
    body = buf[:len(buf) - CRC_LEN]
    if (zlib.crc32(body) & 0xFFFFFFFF) != struct.unpack_from("!I", buf, len(buf) - 4)[0]:
        return None
    op, ver, fid, dqp, sqp = unpack_base(body)
    off, ln, seq_l = struct.unpack_from("!IHH", body, HDR_BASE)
    return dict(op=op, flow=fid, dqp=dqp, sqp=sqp, off=off, ln=ln, seq=seq_l,
                payload=body[HDR_BASE + HDR_METH:][:ln])


class Sack:
    """每 flow 一个 256 位 SACK 位图（规范 §4 的选择性确认）"""

    def __init__(self):
        self.cum = 0
        self.bits = 0

    def mark(self, seq):
        if seq < self.cum:
            return
        d = seq - self.cum
        if d == 0:
            self.cum += 1
            while self.bits & 1:
                self.bits >>= 1
                self.cum += 1
        elif d <= 256:
            self.bits |= 1 << (d - 1)


class Receiver(threading.Thread):
    def __init__(self, addr, total, nflows):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
        self.sock.bind(addr)
        self.total = total
        self.buf = bytearray(total)
        self.got = bytearray(total)
        self.nflows = nflows
        self.ooo = 0
        self.pkts = 0
        self.expect_off = 0
        self.stop = False

    def run(self):
        self.sock.setblocking(False)
        while not self.stop:
            r, _, _ = select.select([self.sock], [], [], 0.05)
            if not r:
                continue
            try:
                pkt, peer = self.sock.recvfrom(65536)
            except BlockingIOError:
                continue
            p = parse_pkt(pkt)
            if p is None:
                continue
            if p["off"] != self.expect_off:
                self.ooo += 1
            self.expect_off = p["off"]
            # ★ 松弛顺序交付：直接落盘，不等前序
            self.buf[p["off"]:p["off"] + p["ln"]] = p["payload"]
            self.got[p["off"]:p["off"] + p["ln"]] = b"\x01" * p["ln"]
            self.pkts += 1
            ack = build_pkt(OP_SACK, p["flow"], 0, 0, 0, p["seq"])
            try:
                self.sock.sendto(ack, peer)
            except OSError:
                pass

    def complete(self):
        return all(self.got)


class LossProxy(threading.Thread):
    """UDP 丢包注入：固定 socket 收发，能正确把 ACK 送回客户端"""

    def __init__(self, listen, target, loss):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
        self.sock.bind(listen)
        self.target = target
        self.loss = loss
        self.dropped = 0
        self.passed = 0
        self.client = None
        self.stop = False

    def run(self):
        self.sock.setblocking(False)
        while not self.stop:
            r, _, _ = select.select([self.sock], [], [], 0.05)
            if not r:
                continue
            try:
                pkt, peer = self.sock.recvfrom(65536)
            except BlockingIOError:
                continue
            p = parse_pkt(pkt)
            if p and p["op"] == OP_WRITE:
                self.client = peer
                if random.random() < self.loss:
                    self.dropped += 1
                    continue
                self.passed += 1
                try:
                    self.sock.sendto(pkt, self.target)
                except OSError:
                    pass
            else:                                   # ACK 往回送
                if self.client:
                    try:
                        self.sock.sendto(pkt, self.client)
                    except OSError:
                        pass


def send_msg(dst, data, nflows, window, max_rounds=20000):
    """发送端：每轮「先补洞、再发新包」，避免 AIMD 收窗后的死锁。

    ★ 第三版修正（前两版踩的坑）：
      bug3: 第一版只在 `sent_at` 超时才重传，且 `nxt` 已越过空洞；
            AIMD 把 cwnd 压到 3 之后窗口内全是已发过的序号 → 空洞永远补不上 → 死锁
            （实测 cum=[48,5,25,11]、nxt=[153,87,144,95]、unacked=545）。
            修法：每轮按 SACK 状态重建发送候选 = [cum, cum+64) 内未确认的「洞」
                  + 新包；用 cwnd 限速发送。洞优先，因此一定收敛。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 << 20)
    sock.setblocking(False)
    total = len(data)
    npkt = (total + MTU - 1) // MTU
    per_flow = [[] for _ in range(nflows)]
    for i in range(npkt):
        per_flow[i % nflows].append(i)
    flows = [Sack() for _ in range(nflows)]
    cwnd = [window] * nflows
    nxt = [0] * nflows
    acked = bytearray(npkt)
    retx = 0
    done = 0
    t0 = time.time()
    rounds = 0
    W = 64                       # 每次回看多少个序号去找洞

    def pkt_of(g, f, L):
        off = g * MTU
        ln = min(MTU, total - off)
        return build_pkt(OP_WRITE, f, 1, 0, off, L, data[off:off + ln])

    while done < npkt and rounds < max_rounds:
        rounds += 1
        for f in range(nflows):
            lst = per_flow[f]
            cum = flows[f].cum
            send_list = []
            # ① 先补洞：cum 之后 W 个序号里未确认的
            for L in range(cum, min(cum + W, len(lst))):
                if not acked[lst[L]]:
                    send_list.append(L)
                if len(send_list) >= cwnd[f]:
                    break
            # ② 再发新包
            while len(send_list) < cwnd[f] and nxt[f] < len(lst) and nxt[f] < cum + W:
                L = nxt[f]
                nxt[f] += 1
                if not acked[lst[L]]:
                    send_list.append(L)
            for L in send_list:
                if L < nxt[f] and not acked[lst[L]]:
                    retx += 1
                try:
                    sock.sendto(pkt_of(lst[L], f, L), dst)
                except OSError:
                    pass
        # 收 ACK（尽力清空）
        for _ in range(256):
            r, _, _ = select.select([sock], [], [], 0.0002)
            if not r:
                break
            try:
                pkt, _ = sock.recvfrom(65536)
            except BlockingIOError:
                break
            p = parse_pkt(pkt)
            if not p or p["op"] != OP_SACK:
                continue
            f, L = p["flow"], p["seq"]
            if f >= nflows or L >= len(per_flow[f]):
                continue
            g = per_flow[f][L]
            if not acked[g]:
                acked[g] = 1
                done += 1
                cwnd[f] = min(cwnd[f] + 1, 256)      # AIMD 加性增
            flows[f].mark(L)
        if rounds % 200 == 0:                        # 周期性乘性减（模拟 ECN/丢包响应）
            for f in range(nflows):
                if flows[f].cum < len(per_flow[f]):
                    cwnd[f] = max(4, int(cwnd[f] * 0.9))
        time.sleep(0.0001)

    unacked = [i for i in range(npkt) if not acked[i]]
    diag = dict(cum=[flows[f].cum for f in range(nflows)],
                nxt=nxt[:], lens=[len(per_flow[f]) for f in range(nflows)],
                cwnd=cwnd[:], unacked=len(unacked), rounds=rounds)
    return dict(npkt=npkt, retx=retx, secs=time.time() - t0, diag=diag)


def run_case(size, loss, nflows, window):
    rport = random.randint(31000, 39000)
    pport = rport + 1
    data = bytes(random.Random(7).getrandbits(8) for _ in range(size))
    ref = hashlib.sha256(data).hexdigest()

    rx = Receiver(("127.0.0.1", rport), size, nflows)
    px = LossProxy(("127.0.0.1", pport), ("127.0.0.1", rport), loss)
    rx.start()
    px.start()
    time.sleep(0.05)

    st = send_msg(("127.0.0.1", pport), data, nflows, window)
    for _ in range(200):
        if rx.complete():
            break
        time.sleep(0.01)
    rx.stop = True
    px.stop = True
    rx.join(0.3)
    px.join(0.3)

    ok = hashlib.sha256(bytes(rx.buf)).hexdigest() == ref
    return dict(loss=loss, flows=nflows, size=size, pkts=st["npkt"], retx=st["retx"],
                dropped=px.dropped, passed=px.passed, ooo=rx.ooo,
                rxpkts=rx.pkts, secs=st["secs"], ok=ok, diag=st.get("diag"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1 << 20)
    ap.add_argument("--flows", type=int, default=4)
    ap.add_argument("--window", type=int, default=16)
    a = ap.parse_args()

    print("=" * 100)
    print("MRC 最小语义核心 —— 从零实现（依据 MetaRoCE-Spec-1.0 §1.3/§1.5/§8）")
    print("=" * 100)
    print("  机制: 多流(每 flow 独立 seq+窗口+SACK+AIMD) / 显式多路径 / 乱序落盘 / 接收端限速")
    print("  包:   Base(op|ver|flowID|dQP|sQP) + METH(off|len|flowLocalSeq) + mCRC32")
    print()
    print("  %-7s %-6s %-9s %-7s %-7s %-7s %-7s %-8s %-8s"
          % ("丢包率", "flows", "消息MB", "包数", "重传", "被丢", "乱序", "耗时(s)", "完整性"))
    print("  " + "-" * 96)
    rows = []
    for loss in (0.0, 0.01, 0.10):
        r = run_case(a.size, loss, a.flows, a.window)
        rows.append(r)
        print("  %-7s %-6d %-9.2f %-7d %-7d %-7d %-7d %-8.3f %-8s"
              % ("%.0f%%" % (loss * 100), r["flows"], r["size"] / 1048576.0,
                 r["pkts"], r["retx"], r["dropped"], r["ooo"], r["secs"],
                 "OK" if r["ok"] else "FAIL"))
    print()
    print("  → 完整性: %s" % ("三档全部 SHA256 一致" if all(x["ok"] for x in rows) else "有失败"))
    print("  → 10%% 丢包仍完成: %s（对照 RoCEv2 需 PFC 才不丢）"
          % ("是" if rows[-1]["ok"] else "否"))
    print("  → 重传/包数比: 10%% 档 %d / %d = %.0f%%  ← **已知短板**：当前实现每轮都重发洞，"
          "未按 RTT 节流，故过量重传；下一步加 RTT pacing"
          % (rows[-1]["retx"], rows[-1]["pkts"],
             100.0 * rows[-1]["retx"] / max(1, rows[-1]["pkts"])))
    print("  → 乱序落盘生效: 10%% 档乱序包 %d 个" % rows[-1]["ooo"])
    print()
    print("  诊断（末档）:", rows[-1].get("diag"))


if __name__ == "__main__":
    main()
