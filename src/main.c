#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include "p2randomv2.h"

#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <ammintrin.h>
#include <x86intrin.h>

static int32_t** tree = NULL;
static int* bound = NULL;

void print128_num(__m128i var)
{
    uint16_t *val = (uint16_t*) &var;
    printf("Numerical: %i %i %i %i %i %i %i %i \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], 
           val[6], val[7]);
}
    
void print_tree(int mxlevel) {
    int i, j;
    for (i = 0; i < mxlevel; ++i) {
        for (j = 0; j < bound[i]; ++j) {
            printf("%d ", tree[i][j]);
        }
        printf("\n");
    }
}

int calcmax (int f[], int mxlevel) {
    int prev = 1;
    int ret = 0;
    int i;
    for (i = 0; i < mxlevel; ++i) {
        ret += prev * (f[i] - 1);
        prev = prev * f[i];
    }
    printf("calcmax: %d\n", ret);
    return ret;
}

int calcmin (int f[], int mxlevel) {
    int ret = 1;
    int i;
    for (i = 1; i < mxlevel; ++i) {
        ret *= f[i];
    }
    printf("calcmin: %d\n", ret);
    return ret;
}

int32_t* gen_rnd_array(int arrlen) {

    int32_t* a = NULL;
    rand32_t *gen = rand32_init(time(NULL));
    //rand32_t *gen = rand32_init(2); // Test
    int32_t *tmp = generate(arrlen, gen);

    posix_memalign((void**)&a, 16, sizeof(int) * arrlen);
    for (int i = 0; i < arrlen; ++i) {
        a[i] = tmp[i];
    }

    free(gen);
    free(tmp);

    return a;
}

int32_t* gen_rnd_array_sn(int arrlen) {

    int32_t* a = NULL;
    rand32_t *gen = rand32_init(time(NULL));
    //rand32_t *gen = rand32_init(1); // Test
    int32_t *tmp = generate_sorted_unique(arrlen, gen);

    posix_memalign((void**)&a, 16, sizeof(int) * arrlen);
    for (int i = 0; i < arrlen; ++i) {
        a[i] = tmp[i];
    }

    free(gen);
    free(tmp);

    return a;
}

void populate_key (int klen, int f[], int mxlevel) {
    int32_t* keys = gen_rnd_array_sn(klen);
    
    int prev = 1;
    int l;
    for (l = mxlevel-1; l >= 0; --l) {
        int litr = 0;
        int ncounter = 0;
        int start = prev - 1;
        int step1 = prev;
        int step2 = prev * 2;
        int i;
        for (i = start; i < klen;) {
            tree[l][litr] = keys[i];
            ++ncounter;
            ++litr;

            if (ncounter == f[l]-1) {
                i += step2;
                ncounter = 0;
            } else {
                i += step1;
            }
        }

        for (i = litr; i < bound[l]; ++i) {
            tree[l][i] = INT_MAX;
        }

        prev *= f[l];
    }

    free(keys);
}

int build_tree (int klen, int f[], int mxlevel) {
    if (posix_memalign((void**)&tree, 16, sizeof(int32_t) * mxlevel) != 0) {
        return -1;
    }

    if (posix_memalign((void**)&bound, 16, sizeof(int) * mxlevel) != 0) {
        return -1;
    }

    int prev = 1;
    int i;
    for (i = mxlevel - 1; i >= 0; --i) {
        int nnum = (klen + prev * (f[i]-1)) / (prev * f[i]);
        bound[i] = nnum * (f[i]-1);
        prev *= f[i];

        //printf("level %d has %d nodes\n", i, nnum);

        if (posix_memalign((void**)(tree + i), 16, sizeof(int32_t) * (f[i]-1) * nnum) != 0) {
            return -1;
        }
    }
    return 0;
}

void prob_sse(int32_t* prob, int32_t* ret, int plen, int* f, int mxlevel) {
    int pi;
    for (pi = 0; pi < plen; pi+=4) {
        int i;

        // load 4 prob keys
        __m128i k = _mm_load_si128((__m128i*) (prob + pi));
        register __m128i k1 = _mm_shuffle_epi32(k, _MM_SHUFFLE(0,0,0,0));
        register __m128i k2 = _mm_shuffle_epi32(k, _MM_SHUFFLE(1,1,1,1));
        register __m128i k3 = _mm_shuffle_epi32(k, _MM_SHUFFLE(2,2,2,2));
        register __m128i k4 = _mm_shuffle_epi32(k, _MM_SHUFFLE(3,3,3,3));

        __m128i delim_A1;
        __m128i delim_B1;
        __m128i delim_C1;
        __m128i delim_D1;

        __m128i delim_A2;
        __m128i delim_B2;
        __m128i delim_C2;
        __m128i delim_D2;

        __m128i delim_A3;
        __m128i delim_B3;
        __m128i delim_C3;
        __m128i delim_D3;

        __m128i delim_A4;
        __m128i delim_B4;
        __m128i delim_C4;
        __m128i delim_D4;

        __m128i cmp_A;
        __m128i cmp_B;
        __m128i cmp_C;
        __m128i cmp_D;

        __m128i tmp1;
        __m128i tmp2;
        __m128i tmp3;

        int msk = 0;
        int res = 0;

        int l1 = 0;
        int l2 = 0;
        int l3 = 0;
        int l4 = 0;

        int ans1 = 0;
        int ans2 = 0;
        int ans3 = 0;
        int ans4 = 0;

        int node_idx1 = 0;
        int node_idx2 = 0;
        int node_idx3 = 0;
        int node_idx4 = 0;

        for (i = 0; i < mxlevel; ++i) {
            if (f[i] == 17) {
                l1 = node_idx1 * 16;
                l2 = node_idx2 * 16;
                l3 = node_idx3 * 16;
                l4 = node_idx4 * 16;

                // load the node of 16 int delimiters
                delim_A1 = _mm_load_si128((__m128i*) &tree[i][l1]);
                delim_B1 = _mm_load_si128((__m128i*) &tree[i][l1+4]);
                delim_C1 = _mm_load_si128((__m128i*) &tree[i][l1+8]);
                delim_D1 = _mm_load_si128((__m128i*) &tree[i][l1+12]);

                delim_A2 = _mm_load_si128((__m128i*) &tree[i][l2]);
                delim_B2 = _mm_load_si128((__m128i*) &tree[i][l2+4]);
                delim_C2 = _mm_load_si128((__m128i*) &tree[i][l2+8]);
                delim_D2 = _mm_load_si128((__m128i*) &tree[i][l2+12]);

                delim_A3 = _mm_load_si128((__m128i*) &tree[i][l3]);
                delim_B3 = _mm_load_si128((__m128i*) &tree[i][l3+4]);
                delim_C3 = _mm_load_si128((__m128i*) &tree[i][l3+8]);
                delim_D3 = _mm_load_si128((__m128i*) &tree[i][l3+12]);

                delim_A4 = _mm_load_si128((__m128i*) &tree[i][l4]);
                delim_B4 = _mm_load_si128((__m128i*) &tree[i][l4+4]);
                delim_C4 = _mm_load_si128((__m128i*) &tree[i][l4+8]);
                delim_D4 = _mm_load_si128((__m128i*) &tree[i][l4+12]);

                // Compare k1
                cmp_A = _mm_cmpgt_epi32(k1, delim_A1);
                cmp_B = _mm_cmpgt_epi32(k1, delim_B1);
                cmp_C = _mm_cmpgt_epi32(k1, delim_C1);
                cmp_D = _mm_cmpgt_epi32(k1, delim_D1);

                tmp1 = _mm_packs_epi32(cmp_A, cmp_B);
                tmp2 = _mm_packs_epi32(cmp_C, cmp_D);
                tmp3 = _mm_packs_epi16(tmp1, tmp2);

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans1 += node_idx1 * 16 + res;
                node_idx1 = node_idx1 * 17 + res;

                // Compare k2
                cmp_A = _mm_cmpgt_epi32(k2, delim_A2);
                cmp_B = _mm_cmpgt_epi32(k2, delim_B2);
                cmp_C = _mm_cmpgt_epi32(k2, delim_C2);
                cmp_D = _mm_cmpgt_epi32(k2, delim_D2);

                tmp1 = _mm_packs_epi32(cmp_A, cmp_B);
                tmp2 = _mm_packs_epi32(cmp_C, cmp_D);
                tmp3 = _mm_packs_epi16(tmp1, tmp2);

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans2 += node_idx2 * 16 + res;
                node_idx2 = node_idx2 * 17 + res;

                // Compare k3
                cmp_A = _mm_cmpgt_epi32(k3, delim_A3);
                cmp_B = _mm_cmpgt_epi32(k3, delim_B3);
                cmp_C = _mm_cmpgt_epi32(k3, delim_C3);
                cmp_D = _mm_cmpgt_epi32(k3, delim_D3);

                tmp1 = _mm_packs_epi32(cmp_A, cmp_B);
                tmp2 = _mm_packs_epi32(cmp_C, cmp_D);
                tmp3 = _mm_packs_epi16(tmp1, tmp2);

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans3 += node_idx3 * 16 + res;
                node_idx3 = node_idx3 * 17 + res;

                // Compare k4
                cmp_A = _mm_cmpgt_epi32(k4, delim_A4);
                cmp_B = _mm_cmpgt_epi32(k4, delim_B4);
                cmp_C = _mm_cmpgt_epi32(k4, delim_C4);
                cmp_D = _mm_cmpgt_epi32(k4, delim_D4);

                tmp1 = _mm_packs_epi32(cmp_A, cmp_B);
                tmp2 = _mm_packs_epi32(cmp_C, cmp_D);
                tmp3 = _mm_packs_epi16(tmp1, tmp2);

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans4 += node_idx4 * 16 + res;
                node_idx4 = node_idx4 * 17 + res;

            }  else if (f[i] == 9) {
                l1 = node_idx1 * 8;
                l2 = node_idx2 * 8;
                l3 = node_idx3 * 8;
                l4 = node_idx4 * 8;

                // load the node of 8 int delimiters
                delim_A1 = _mm_load_si128((__m128i*) &tree[i][l1]);
                delim_B1 = _mm_load_si128((__m128i*) &tree[i][l1+4]);

                delim_A2 = _mm_load_si128((__m128i*) &tree[i][l2]);
                delim_B2 = _mm_load_si128((__m128i*) &tree[i][l2+4]);

                delim_A3 = _mm_load_si128((__m128i*) &tree[i][l3]);
                delim_B3 = _mm_load_si128((__m128i*) &tree[i][l3+4]);

                delim_A4 = _mm_load_si128((__m128i*) &tree[i][l4]);
                delim_B4 = _mm_load_si128((__m128i*) &tree[i][l4+4]);

                // Compare k1
                cmp_A = _mm_cmpgt_epi32(k1, delim_A1);
                cmp_B = _mm_cmpgt_epi32(k1, delim_B1);

                tmp1 = _mm_packs_epi32(cmp_A, cmp_B);
                tmp3 = _mm_packs_epi16(tmp1, _mm_setzero_si128());

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans1 += node_idx1 * 8 + res;
                node_idx1 = node_idx1 * 9 + res;

                // Compare k2
                cmp_A = _mm_cmpgt_epi32(k2, delim_A2);
                cmp_B = _mm_cmpgt_epi32(k2, delim_B2);

                tmp1 = _mm_packs_epi32(cmp_A, cmp_B);
                tmp3 = _mm_packs_epi16(tmp1, _mm_setzero_si128());

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans2 += node_idx2 * 8 + res;
                node_idx2 = node_idx2 * 9 + res;

                // Compare k3
                cmp_A = _mm_cmpgt_epi32(k3, delim_A3);
                cmp_B = _mm_cmpgt_epi32(k3, delim_B3);

                tmp1 = _mm_packs_epi32(cmp_A, cmp_B);
                tmp3 = _mm_packs_epi16(tmp1, _mm_setzero_si128());

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans3 += node_idx3 * 8 + res;
                node_idx3 = node_idx3 * 9 + res;

                // Compare k4
                cmp_A = _mm_cmpgt_epi32(k4, delim_A4);
                cmp_B = _mm_cmpgt_epi32(k4, delim_B4);

                tmp1 = _mm_packs_epi32(cmp_A, cmp_B);
                tmp3 = _mm_packs_epi16(tmp1, _mm_setzero_si128());

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans4 += node_idx4 * 8 + res;
                node_idx4 = node_idx4 * 9 + res;

            }  else if (f[i] == 5) {
                l1 = node_idx1 * 4;
                l2 = node_idx2 * 4;
                l3 = node_idx3 * 4;
                l4 = node_idx4 * 4;

                // load the node of 4 int delimiters
                delim_A1 = _mm_load_si128((__m128i*) &tree[i][l1]);
                delim_A2 = _mm_load_si128((__m128i*) &tree[i][l2]);
                delim_A3 = _mm_load_si128((__m128i*) &tree[i][l3]);
                delim_A4 = _mm_load_si128((__m128i*) &tree[i][l4]);

                // Compare k1
                cmp_A = _mm_cmpgt_epi32(k1, delim_A1);

                tmp1 = _mm_packs_epi32(cmp_A, _mm_setzero_si128());
                tmp3 = _mm_packs_epi16(tmp1, _mm_setzero_si128());

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans1 += node_idx1 * 4 + res;
                node_idx1 = node_idx1 * 5 + res;

                // Compare k2
                cmp_A = _mm_cmpgt_epi32(k2, delim_A2);

                tmp1 = _mm_packs_epi32(cmp_A, _mm_setzero_si128());
                tmp3 = _mm_packs_epi16(tmp1, _mm_setzero_si128());

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans2 += node_idx2 * 4 + res;
                node_idx2 = node_idx2 * 5 + res;

                // Compare k3
                cmp_A = _mm_cmpgt_epi32(k3, delim_A3);

                tmp1 = _mm_packs_epi32(cmp_A, _mm_setzero_si128());
                tmp3 = _mm_packs_epi16(tmp1, _mm_setzero_si128());

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans3 += node_idx3 * 4 + res;
                node_idx3 = node_idx3 * 5 + res;

                // Compare k4
                cmp_A = _mm_cmpgt_epi32(k4, delim_A4);

                tmp1 = _mm_packs_epi32(cmp_A, _mm_setzero_si128());
                tmp3 = _mm_packs_epi16(tmp1, _mm_setzero_si128());

                msk = _mm_movemask_epi8(tmp3);
                msk = msk << 1;

                res = _bit_scan_reverse(msk);
                ans4 += node_idx4 * 4 + res;
                node_idx4 = node_idx4 * 5 + res;
            }
        }

        if (pi < plen) {
            ret[pi] = ans1;
        }
        if (pi + 1 < plen) {
            ret[pi+1] = ans2;
        }
        if (pi + 2 < plen) {
            ret[pi+2] = ans3;
        }
        if (pi + 3 < plen) {
            ret[pi+3] = ans4;
        }
    }
}
 
void prob_binary(int32_t* prob, int32_t* ret, int plen, int* f, int mxlevel) {
    int pi;
    for (pi = 0; pi < plen; ++pi) {
        int key = prob[pi];
        int node_idx = 0; /* root level is always 0 */
        int l, r;
        int ans = 0;
        int i;
        for (i = 0; i < mxlevel; ++i) {
            l = node_idx * (f[i] - 1);
            r = node_idx * (f[i] - 1) + f[i] - 2;
            while (l <= r) {
                int mid = l + (r - l) / 2;
                if (key <= tree[i][mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            }
            
            //node_idx = node_idx * f[i] + l - node_idx * (f[i] - 1);
            node_idx = l + node_idx;
            ans += l;
        }
        ret[pi] = ans;
    }
}
 
void prob_hardcode(int32_t* prob, int32_t* ret, int plen) {
    
    register __m128i lvl_0_A = _mm_load_si128((__m128i*) &tree[0][0]);
    register __m128i lvl_0_B = _mm_load_si128((__m128i*) &tree[0][4]);

    int i;
    for (i = 0; i < plen; i+=4) {

        __m128i k = _mm_load_si128((__m128i*) (prob + i));

        register __m128i k1 = _mm_shuffle_epi32(k, _MM_SHUFFLE(0,0,0,0));
        register __m128i k2 = _mm_shuffle_epi32(k, _MM_SHUFFLE(1,1,1,1));
        register __m128i k3 = _mm_shuffle_epi32(k, _MM_SHUFFLE(2,2,2,2));
        register __m128i k4 = _mm_shuffle_epi32(k, _MM_SHUFFLE(3,3,3,3));
    
        __m128i cmp_0_1_A = _mm_cmpgt_epi32(k1, lvl_0_A);
        __m128i cmp_0_1_B = _mm_cmpgt_epi32(k1, lvl_0_B);

        __m128i cmp_0_2_A = _mm_cmpgt_epi32(k2, lvl_0_A);
        __m128i cmp_0_2_B = _mm_cmpgt_epi32(k2, lvl_0_B);

        __m128i cmp_0_3_A = _mm_cmpgt_epi32(k3, lvl_0_A);
        __m128i cmp_0_3_B = _mm_cmpgt_epi32(k3, lvl_0_B);

        __m128i cmp_0_4_A = _mm_cmpgt_epi32(k4, lvl_0_A);
        __m128i cmp_0_4_B = _mm_cmpgt_epi32(k4, lvl_0_B);

        __m128i cmp_0_1 = _mm_packs_epi32(cmp_0_1_A, cmp_0_1_B);
        cmp_0_1 = _mm_packs_epi16(cmp_0_1, _mm_setzero_si128());

        __m128i cmp_0_2 = _mm_packs_epi32(cmp_0_2_A, cmp_0_2_B);
        cmp_0_2 = _mm_packs_epi16(cmp_0_2, _mm_setzero_si128());

        __m128i cmp_0_3 = _mm_packs_epi32(cmp_0_3_A, cmp_0_3_B);
        cmp_0_3 = _mm_packs_epi16(cmp_0_3, _mm_setzero_si128());

        __m128i cmp_0_4 = _mm_packs_epi32(cmp_0_4_A, cmp_0_4_B);
        cmp_0_4 = _mm_packs_epi16(cmp_0_4, _mm_setzero_si128());

        int msk_0_1 = _mm_movemask_epi8(cmp_0_1);
        msk_0_1 = msk_0_1 << 1;
        int msk_0_2 = _mm_movemask_epi8(cmp_0_2);
        msk_0_2 = msk_0_2 << 1;
        int msk_0_3 = _mm_movemask_epi8(cmp_0_3);
        msk_0_3 = msk_0_3 << 1;
        int msk_0_4 = _mm_movemask_epi8(cmp_0_4);
        msk_0_4 = msk_0_4 << 1;

        int res_0_1 = _bit_scan_reverse(msk_0_1);
        int res_0_2 = _bit_scan_reverse(msk_0_2);
        int res_0_3 = _bit_scan_reverse(msk_0_3);
        int res_0_4 = _bit_scan_reverse(msk_0_4);

        int ans1 = res_0_1;
        int ans2 = res_0_2;
        int ans3 = res_0_3;
        int ans4 = res_0_4;

        int node_idx1 = res_0_1;
        int node_idx2 = res_0_2;
        int node_idx3 = res_0_3;
        int node_idx4 = res_0_4;

        int l1 = node_idx1 * 4;
        int l2 = node_idx2 * 4;
        int l3 = node_idx3 * 4;
        int l4 = node_idx4 * 4;

        __m128i lvl_1_A1 = _mm_load_si128((__m128i*) &tree[1][l1]);
        __m128i lvl_1_A2 = _mm_load_si128((__m128i*) &tree[1][l2]);
        __m128i lvl_1_A3 = _mm_load_si128((__m128i*) &tree[1][l3]);
        __m128i lvl_1_A4 = _mm_load_si128((__m128i*) &tree[1][l4]);

        cmp_0_1_A = _mm_cmpgt_epi32(k1, lvl_1_A1);
        cmp_0_2_A = _mm_cmpgt_epi32(k2, lvl_1_A2);
        cmp_0_3_A = _mm_cmpgt_epi32(k3, lvl_1_A3);
        cmp_0_4_A = _mm_cmpgt_epi32(k4, lvl_1_A4);

        cmp_0_1 = _mm_packs_epi32(cmp_0_1_A, _mm_setzero_si128());
        cmp_0_2 = _mm_packs_epi32(cmp_0_2_A, _mm_setzero_si128());
        cmp_0_3 = _mm_packs_epi32(cmp_0_3_A, _mm_setzero_si128());
        cmp_0_4 = _mm_packs_epi32(cmp_0_4_A, _mm_setzero_si128());

        cmp_0_1 = _mm_packs_epi16(cmp_0_1, _mm_setzero_si128());
        cmp_0_2 = _mm_packs_epi16(cmp_0_2, _mm_setzero_si128());
        cmp_0_3 = _mm_packs_epi16(cmp_0_3, _mm_setzero_si128());
        cmp_0_4 = _mm_packs_epi16(cmp_0_4, _mm_setzero_si128());

        msk_0_1 = _mm_movemask_epi8(cmp_0_1);
        msk_0_1 = msk_0_1 << 1;
        msk_0_2 = _mm_movemask_epi8(cmp_0_2);
        msk_0_2 = msk_0_2 << 1;
        msk_0_3 = _mm_movemask_epi8(cmp_0_3);
        msk_0_3 = msk_0_3 << 1;
        msk_0_4 = _mm_movemask_epi8(cmp_0_4);
        msk_0_4 = msk_0_4 << 1;

        res_0_1 = _bit_scan_reverse(msk_0_1);
        res_0_2 = _bit_scan_reverse(msk_0_2);
        res_0_3 = _bit_scan_reverse(msk_0_3);
        res_0_4 = _bit_scan_reverse(msk_0_4);
        
        ans1 += node_idx1 * 4 + res_0_1;
        ans2 += node_idx2 * 4 + res_0_2;
        ans3 += node_idx3 * 4 + res_0_3;
        ans4 += node_idx4 * 4 + res_0_4;

        node_idx1 = node_idx1 * 5 + res_0_1;
        node_idx2 = node_idx2 * 5 + res_0_2;
        node_idx3 = node_idx3 * 5 + res_0_3;
        node_idx4 = node_idx4 * 5 + res_0_4;

        l1 = node_idx1 * 8;
        l2 = node_idx2 * 8;
        l3 = node_idx3 * 8;
        l4 = node_idx4 * 8;

        __m128i lvl_2_A1 = _mm_load_si128((__m128i*) &tree[2][l1]);
        __m128i lvl_2_B1 = _mm_load_si128((__m128i*) &tree[2][l1+4]);
        __m128i lvl_2_A2 = _mm_load_si128((__m128i*) &tree[2][l2]);
        __m128i lvl_2_B2 = _mm_load_si128((__m128i*) &tree[2][l2+4]);
        __m128i lvl_2_A3 = _mm_load_si128((__m128i*) &tree[2][l3]);
        __m128i lvl_2_B3 = _mm_load_si128((__m128i*) &tree[2][l3+4]);
        __m128i lvl_2_A4 = _mm_load_si128((__m128i*) &tree[2][l4]);
        __m128i lvl_2_B4 = _mm_load_si128((__m128i*) &tree[2][l4+4]);

        cmp_0_1_A = _mm_cmpgt_epi32(k1, lvl_2_A1);
        cmp_0_1_B = _mm_cmpgt_epi32(k1, lvl_2_B1);
        cmp_0_2_A = _mm_cmpgt_epi32(k2, lvl_2_A2);
        cmp_0_2_B = _mm_cmpgt_epi32(k2, lvl_2_B2);
        cmp_0_3_A = _mm_cmpgt_epi32(k3, lvl_2_A3);
        cmp_0_3_B = _mm_cmpgt_epi32(k3, lvl_2_B3);
        cmp_0_4_A = _mm_cmpgt_epi32(k4, lvl_2_A4);
        cmp_0_4_B = _mm_cmpgt_epi32(k4, lvl_2_B4);

        cmp_0_1 = _mm_packs_epi32(cmp_0_1_A, cmp_0_1_B);
        cmp_0_2 = _mm_packs_epi32(cmp_0_2_A, cmp_0_2_B);
        cmp_0_3 = _mm_packs_epi32(cmp_0_3_A, cmp_0_3_B);
        cmp_0_4 = _mm_packs_epi32(cmp_0_4_A, cmp_0_4_B);

        cmp_0_1 = _mm_packs_epi16(cmp_0_1, _mm_setzero_si128());
        cmp_0_2 = _mm_packs_epi16(cmp_0_2, _mm_setzero_si128());
        cmp_0_3 = _mm_packs_epi16(cmp_0_3, _mm_setzero_si128());
        cmp_0_4 = _mm_packs_epi16(cmp_0_4, _mm_setzero_si128());

        msk_0_1 = _mm_movemask_epi8(cmp_0_1);
        msk_0_1 = msk_0_1 << 1;
        msk_0_2 = _mm_movemask_epi8(cmp_0_2);
        msk_0_2 = msk_0_2 << 1;
        msk_0_3 = _mm_movemask_epi8(cmp_0_3);
        msk_0_3 = msk_0_3 << 1;
        msk_0_4 = _mm_movemask_epi8(cmp_0_4);
        msk_0_4 = msk_0_4 << 1;

        res_0_1 = _bit_scan_reverse(msk_0_1);
        res_0_2 = _bit_scan_reverse(msk_0_2);
        res_0_3 = _bit_scan_reverse(msk_0_3);
        res_0_4 = _bit_scan_reverse(msk_0_4);

        ans1 += node_idx1 * 8 + res_0_1;
        ans2 += node_idx2 * 8 + res_0_2;
        ans3 += node_idx3 * 8 + res_0_3;
        ans4 += node_idx4 * 8 + res_0_4;

        if (i < plen) {
            ret[i] = ans1;
        }
        if (i + 1 < plen) {
            ret[i+1] = ans2;
        }
        if (i + 2 < plen) {
            ret[i+2] = ans3;
        }
        if (i + 3 < plen) {
            ret[i+3] = ans4;
        }
    }
}

void print_result(int32_t* prob, int32_t* result, int plen) {

    printf("\n\nProbing Result:\n\n");
    int i;
    for (i = 0; i < plen; ++i) {
        printf("%d %d\n", prob[i], result[i]);
    }
}

int main (int argc, char** argv) {

    if (argc < 4) {
        printf("Expect more args\n");
        return -1;
    }

    int k = 0;
    int p = 0;
    int mxlevel = argc - 3;
    int* f = NULL;
    if (posix_memalign((void**)&f, 16, sizeof(int) * mxlevel) != 0) {
        return -1;
    }

    k = atoi(argv[1]);
    //printf("k=%d\n", k);

    p = atoi(argv[2]);
    //printf("p=%d\n", p);
    
    int i;
    for (i = 0; i < mxlevel; ++i) {
        if (atoi(argv[i+3]) != 5 && atoi(argv[i+3]) != 9 && atoi(argv[i+3]) != 17) {
            printf("In valid fanout.\nOnly 5, 9, and 17 are allowed in part 2.\n");
            free(f);
            return -1;
        }
        f[i] = atoi(argv[i+3]);
        //printf("f[%d]=%d ", i, f[i]);
        // if (f[i] < 2 || f[i] > 17) {
        //     printf("Invalid fanout\n");
        //     free(f);
        //     return -1;
        // }
    }
    printf("\n");

    if (k > calcmax(f, mxlevel)) {
        printf("Too much build key\n");
        free(f);
        return -1;
    }

    if (k < calcmin(f, mxlevel)) {
        printf("Too few build key\n");
        free(f);
        return -1;
    }

    if (build_tree(k, f, mxlevel) < 0) {
        return -1;
    }

    populate_key(k, f, mxlevel);

    //print_tree(mxlevel);

    int32_t* prob = gen_rnd_array(p);
    int32_t* result = NULL;
    if (posix_memalign((void**)&result, 16, sizeof(int32_t) * p) != 0) {
        return -1;
    }

    struct timeval stv;
    struct timeval etv;
    gettimeofday(&stv, NULL);

    if (mxlevel == 3 && f[0] == 9 && f[1] == 5 && f[2] == 9) {
        //printf("Go to hardcode version.\n");
        prob_hardcode(prob, result, p);
    } else {
        prob_sse(prob, result, p, f, mxlevel);
    }

    gettimeofday(&etv, NULL);

    print_result(prob, result, p);
    printf("\n\nTime: %ld microseconds\n", (etv.tv_sec-stv.tv_sec) * 1000000L + (etv.tv_usec-stv.tv_usec));

    free(prob);
    free(result);
    free(f);

    for (i = 0; i < mxlevel; ++i) {
        free(tree[i]);
    }
    free(tree);
    free(bound);

    return 0;
}
