/*
 * mx450_p1.c — P1：用 kprobe 绕过 kallsyms_lookup_name 未导出的限制
 *
 * 原理：kallsyms_lookup_name 自 Linux 5.7 起不再 EXPORT_SYMBOL，
 *      直接调用会链接失败。但 register_kprobe() 会用符号名解析出地址，
 *      我们把 kp.addr 读出来即可得到真实函数地址，再转型调用。
 *
 * 本模块只读：仅查询符号地址并打印，不修改任何内核/GPU 状态。
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/kprobes.h>
#include <linux/string.h>

typedef unsigned long (*kln_t)(const char *name);

/* 待查询的符号：NVIDIA 内存管理内部接口 */
static const char *targets[] = {
    "os_map_kernel_space",
    "nv_al_add_mapping",
    "nv_alloc_mem",
    "nv_free_mem",
    NULL
};

static int __init mx450_p1_init(void)
{
    struct kprobe kp;
    kln_t kln;
    unsigned long kln_addr;
    int ret, i;

    memset(&kp, 0, sizeof(kp));
    kp.symbol_name = "kallsyms_lookup_name";

    ret = register_kprobe(&kp);
    if (ret < 0) {
        pr_err("mx450_p1: register_kprobe failed: %d\n", ret);
        return ret;
    }

    kln_addr = (unsigned long)kp.addr;
    unregister_kprobe(&kp);

    pr_info("mx450_p1: kallsyms_lookup_name resolved @ 0x%lx\n", kln_addr);

    if (!kln_addr) {
        pr_err("mx450_p1: kprobe returned null address\n");
        return -ENOENT;
    }

    kln = (kln_t)kln_addr;

    for (i = 0; targets[i]; i++) {
        unsigned long a;
        a = kln(targets[i]);
        if (a)
            pr_info("mx450_p1: %-28s = 0x%lx  [FOUND]\n", targets[i], a);
        else
            pr_info("mx450_p1: %-28s = (not present)\n", targets[i]);
    }

    pr_info("mx450_p1: probe complete\n");
    return 0;
}

static void __exit mx450_p1_exit(void)
{
    pr_info("mx450_p1: unloaded\n");
}

module_init(mx450_p1_init);
module_exit(mx450_p1_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LYLNAS Research");
MODULE_DESCRIPTION("MX450 P1 - kallsyms_lookup_name bypass probe");
