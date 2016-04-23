#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>


typedef struct {
	size_t index;
	uint32_t num[625];
} rand32_t;

rand32_t *rand32_init(uint32_t x)
{
	rand32_t *s = malloc(sizeof(rand32_t));
	uint32_t *n = s->num;
	size_t i = 1;
	n[0] = x;
	do {
		x = 0x6c078965 * (x ^ (x >> 30));
		n[i] = x;
	} while (++i != 624);
	s->index = i;
	return s;
}

uint32_t rand32_next(rand32_t *s)
{
	uint32_t x, *n = s->num;
	size_t i = s->index;
	if (i == 624) {
		i = 0;
		do {
			x = (n[i] & 0x80000000) + (n[i + 1] & 0x7fffffff);
			n[i] = (n[i + 397] ^ (x >> 1)) ^ (0x9908b0df & -(x & 1));
		} while (++i != 227);
		n[624] = n[0];
		do {
			x = (n[i] & 0x80000000) + (n[i + 1] & 0x7fffffff);
			n[i] = (n[i - 227] ^ (x >> 1)) ^ (0x9908b0df & -(x & 1));
		} while (++i != 624);
		i = 0;
	}
	x = n[i];
	x ^= (x >> 11);
	x ^= (x <<  7) & 0x9d2c5680;
	x ^= (x << 15) & 0xefc60000;
	x ^= (x >> 18);
	s->index = i + 1;
	return x;
}

int int32_cmp(const void *x, const void *y)
{
	int32_t a = * (const int*) x;
	int32_t b = * (const int*) y;
	return a < b ? -1 : a > b ? 1 : 0;
}

int32_t *generate(size_t n, rand32_t *gen)
{
	size_t i;
	int32_t *a = malloc(n << 2);
	for (i = 0 ; i != n ; ++i)
		a[i] = rand32_next(gen);
	return a;
}

int32_t *generate_sorted_unique(size_t n, rand32_t *gen)
{
	size_t i = 0;
	size_t m = n / 0.7;
	uint8_t z = 0;
	uint32_t *a = malloc(n << 2);
	uint32_t *b = calloc(m, 4);
	while (i != n) {
		uint32_t k = rand32_next(gen);
		if (k != 0) {
			size_t h = (uint32_t) (k * 0x9e3779b1);
			h = (h * (uint64_t) m) >> 32;
			while (b[h] != k) {
				if (b[h] == 0) {
					b[h] = a[i++] = k;
					break;
				}
				if (++h == m) h = 0;
			}
		} else if (z == 0) {
			a[i++] = 0;
			z = 1;
		}
	}
	free(b);
	qsort(a, n, 4, int32_cmp);
	return (int32_t*) a;
}

void ratio_per_bit(const int32_t *a, size_t n)
{
    /*
	size_t i, j, *c = calloc(32, sizeof(size_t));
	for (i = 0 ; i != n ; ++i) {
		int32_t x = a[i];
		for (j = 0 ; j != 32 ; ++j)
			c[j] += (a[i] >> j) & 1;
	}
	for (j = 0 ; j != 32 ; ++j)
		fprintf(stderr, "%2ld: %.2f%%\n", j + 1, c[j] * 100.0 / n);
	free(c);
    */
}

/*
int main(int argc, char **argv)
{
	rand32_t *gen = rand32_init(time(NULL));
	size_t i, n = argc > 1 ? atoll(argv[1]) : 10;
	int32_t *a = generate_sorted_unique(n, gen);
	free(gen);
	for (i = 1 ; i < n ; ++i)
		assert(a[i - 1] < a[i]);
	ratio_per_bit(a, n);
	free(a);
	return EXIT_SUCCESS;
}
*/
