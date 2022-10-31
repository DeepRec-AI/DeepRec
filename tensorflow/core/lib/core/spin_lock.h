#ifndef THIRD_PARTY_TENSORFLOW_CORE_LIB_CORE_SPINLOCK_H
#define THIRD_PARTY_TENSORFLOW_CORE_LIB_CORE_SPINLOCK_H

namespace tensorflow {
namespace {
/* Compile read-write barrier */
#define mem_barrier() asm volatile("": : :"memory")

/* Pause instruction to prevent excess processor bus usage */ 
#if defined(__x86_64)
#define cpu_relax() asm volatile("pause\n": : :"memory")
#else
#define cpu_relax() asm volatile("yield\n": : :"memory")
#endif

# define __ASM_FORM(x)  " " #x " "
# define __ASM_SEL(a,b) __ASM_FORM(a)
#define _ASM_ALIGN  __ASM_SEL(.balign 4, .balign 8)
#define _ASM_PTR  __ASM_SEL(.long, .quad)

#define LOCK_PREFIX \
      ".section .smp_locks,\"a\"\n" \
    _ASM_ALIGN "\n"     \
    _ASM_PTR "661f\n" /* address */ \
    ".previous\n"     \
    "661:\n\tlock; "
#define LOCK_PREFIX \
      ".section .smp_locks,\"a\"\n" \
    _ASM_ALIGN "\n"     \
    _ASM_PTR "661f\n" /* address */ \
    ".previous\n"     \
    "661:\n\tlock; "

/* Atomic exchange (of various sizes) */
static inline unsigned long xchg_64(void *ptr, unsigned long x)
{
#if defined(__x86_64)
  asm volatile("xchgq %0,%1"
      :"=r" ((unsigned long) x)
      :"m" (*(volatile long *)ptr), "0" ((unsigned long) x)
      :"memory");
#else
  x =  __atomic_exchange_n((unsigned long *)ptr, x, __ATOMIC_SEQ_CST);
#endif

  return x;
}

static void lock_impl(unsigned long* lock) {
  while (xchg_64((void*)lock, 1)) {
    while (*lock) cpu_relax();
  }
}

static void unlock_impl(unsigned long* lock) {
  mem_barrier();
  *lock = 0;
}
}

class spin_lock {
public:
  spin_lock() = default;
  spin_lock(const spin_lock&) = delete;
  spin_lock& operator= (const spin_lock&) = delete;

  void lock() {
    lock_impl(&lock_);
  }

  void unlock() {
    unlock_impl(&lock_);
  }

private:
  unsigned long lock_ = 0;
};

}
#endif
