/*
 * mx450_p2.c — P2 第一步：验证从 /proc/kallsyms 取得的符号地址是否有效
 *
 * 设计原则：
 *  1. 地址由用户态读 /proc/kallsyms 得到，通过 module_param 传入
 *  2. 全程用 copy_from_kernel_nofault() 读内核内存 —— 地址无效只返回错误，不会 oops
 *  3. 不调用目标函数，只读取其机器码做指纹识别
 *
 * 用法：
 *  insmod mx450_p2.ko target_addr=0xffffffffc2c13bc0
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/uaccess.h>

static unsigned long target_addr = 0;
module_param(target_addr, ulong, 0444);
MODULE_PARM_DESC(target_addr, "Kernel symbol address (from /proc/kallsyms)");

static int __init mx450_p2_init(void)
{
    unsigned char buf[32];
    int ret, i;
    char hex[3 * 32 + 1];
    int pos = 0;

    if (!target_addr) {
        pr_err("mx450_p2: target_addr not set\n");
        return -EINVAL;
    }

    pr_info("mx450_p2: probing 0x%lx\n", target_addr);

    ret = copy_from_kernel_nofault(buf, (void *)target_addr, sizeof(buf));
    if (ret) {
        pr_err("mx450_p2: copy_from_kernel_nofault failed: %d (address likely invalid)\n", ret);
        return -EFAULT;
    }

    for (i = 0; i < (int)sizeof(buf); i++) {
        pos += scnprintf(hex + pos, sizeof(hex) - pos, "%02x ", buf[i]);
    }
    pr_info("mx450_p2: bytes: %s\n", hex);

    /* 常见 x86-64 函数序言指纹 */
    if (buf[0] == 0xf3 && buf[1] == 0x0f && buf[2] == 0x1e && buf[3] == 0xfa)
        pr_info("mx450_p2: signature = endbr64 (CET/BTI 函数入口) [VALID]\n");
    else if (buf[0] == 0x55)
        pr_info("mx450_p2: signature = push rbp (传统函数序言) [VALID]\n");
    else if (buf[0] == 0x48 && buf[1] == 0x8b)
        pr_info("mx450_p2: signature = mov r64,r/m64 [possible mid-function]\n");
    else
        pr_info("mx450_p2: signature unrecognized (first byte 0x%02x)\n", buf[0]);

    pr_info("mx450_p2: probe done, address is readable\n");
    return 0;
}

static void __exit mx450_p2_exit(void)
{
    pr_info("mx450_p2: unloaded\n");
}

module_init(mx450_p2_init);
module_exit(mx450_p2_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LYLNAS Research");
MODULE_DESCRIPTION("MX450 P2 - kernel symbol address probe (read-only, nofault)");
