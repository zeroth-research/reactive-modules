/*
 * Sanity: 1-D decrement.
 *
 * The simplest possible terminating loop. Used as a smoke test that
 * the entire pipeline (translate → train → verify) is wired up.
 *
 * Expected: VERIFIED with V(x) = ReLU(x).
 */

extern int __VERIFIER_nondet_int(void);

int main() {
    int x = __VERIFIER_nondet_int();
    while (x > 0) {
        x = x - 1;
    }
    return 0;
}
