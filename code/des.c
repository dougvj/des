/****************************************************************************
   (C) 2012 Doug Johnson

    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*****************************************************************************/

#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
//#include <assert.h>
#include <sys/time.h>

#ifndef max
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


#ifdef OPENCL
    #include <CL/cl.h>
#endif

//Enable or disable debugging output here

//#define DEBUG_EXTRACTOR
//#define DEBUG_PERMUTER
//#define DEBUG_ROTATE
//#define DEBUG_KEYGEN
//#define DEBUG_ROUNDS
//#define DEBUG_MANGLER
//#define DEBUG_SBOX

static uint32_t sbox[8][4][16] = {                      
//S1  0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
    {{0xe, 0x4, 0xd, 0x1, 0x2, 0xf, 0xb, 0x8, 0x3, 0xa, 0x6, 0xc, 0x5, 0x9, 0x0, 0x7},
     {0x0, 0xf, 0x7, 0x4, 0xe, 0x2, 0xd, 0x1, 0xa, 0x6, 0xc, 0xb, 0x9, 0x5, 0x3, 0x8},
     {0x4, 0x1, 0xe, 0x8, 0xd, 0x6, 0x2, 0xb, 0xf, 0xc, 0x9, 0x7, 0x3, 0xa, 0x5, 0x0},
     {0xf, 0xc, 0x8, 0x2, 0x4, 0x9, 0x1, 0x7, 0x5, 0xb, 0x3, 0xe, 0xa, 0x0, 0x6, 0xd}},
                           
//S2 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
    {{0xf, 0x1, 0x8, 0xe, 0x6, 0xb, 0x3, 0x4, 0x9, 0x7, 0x2, 0xd, 0xc, 0x0, 0x5, 0xa},
     {0x3, 0xd, 0x4, 0x7, 0xf, 0x2, 0x8, 0xe, 0xc, 0x0, 0x1, 0xa, 0x6, 0x9, 0xb, 0x5},
     {0x0, 0xe, 0x7, 0xb, 0xa, 0x4, 0xd, 0x1, 0x5, 0x8, 0xc, 0x6, 0x9, 0x3, 0x2, 0xf},
     {0xd, 0x8, 0xa, 0x1, 0x3, 0xf, 0x4, 0x2, 0xb, 0x6, 0x7, 0xc, 0x0, 0x5, 0xe, 0x9}},
                           
//S3 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
   {{0xa, 0x0, 0x9, 0xe, 0x6, 0x3, 0xf, 0x5, 0x1, 0xd, 0xc, 0x7, 0xb, 0x4, 0x2, 0x8},
    {0xd, 0x7, 0x0, 0x9, 0x3, 0x4, 0x6, 0xa, 0x2, 0x8, 0x5, 0xe, 0xc, 0xb, 0xf, 0x1},
    {0xd, 0x6, 0x4, 0x9, 0x8, 0xf, 0x3, 0x0, 0xb, 0x1, 0x2, 0xc, 0x5, 0xa, 0xe, 0x7},
    {0x1, 0xa, 0xd, 0x0, 0x6, 0x9, 0x8, 0x7, 0x4, 0xf, 0xe, 0x3, 0xb, 0x5, 0x2, 0xc}},
                           
//S4 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
   {{0x7, 0xd, 0xe, 0x3, 0x0, 0x6, 0x9, 0xa, 0x1, 0x2, 0x8, 0x5, 0xb, 0xc, 0x4, 0xf},
    {0xd, 0x8, 0xb, 0x5, 0x6, 0xf, 0x0, 0x3, 0x4, 0x7, 0x2, 0xc, 0x1, 0xa, 0xe, 0x9},
    {0xa, 0x6, 0x9, 0x0, 0xc, 0xb, 0x7, 0xd, 0xf, 0x1, 0x3, 0xe, 0x5, 0x2, 0x8, 0x4},
    {0x3, 0xf, 0x0, 0x6, 0xa, 0x1, 0xd, 0x8, 0x9, 0x4, 0x5, 0xb, 0xc, 0x7, 0x2, 0xe}},
                           
//S5 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
   {{0x2, 0xc, 0x4, 0x1, 0x7, 0xa, 0xb, 0x6, 0x8, 0x5, 0x3, 0xf, 0xd, 0x0, 0xe, 0x9},
    {0xe, 0xb, 0x2, 0xc, 0x4, 0x7, 0xd, 0x1, 0x5, 0x0, 0xf, 0xa, 0x3, 0x9, 0x8, 0x6},
    {0x4, 0x2, 0x1, 0xb, 0xa, 0xd, 0x7, 0x8, 0xf, 0x9, 0xc, 0x5, 0x6, 0x3, 0x0, 0xe},
    {0xb, 0x8, 0xc, 0x7, 0x1, 0xe, 0x2, 0xd, 0x6, 0xf, 0x0, 0x9, 0xa, 0x4, 0x5, 0x3}},
                           
//S6 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
   {{0xc, 0x1, 0xa, 0xf, 0x9, 0x2, 0x6, 0x8, 0x0, 0xd, 0x3, 0x4, 0xe, 0x7, 0x5, 0xb},
    {0xa, 0xf, 0x4, 0x2, 0x7, 0xc, 0x9, 0x5, 0x6, 0x1, 0xd, 0xe, 0x0, 0xb, 0x3, 0x8},
    {0x9, 0xe, 0xf, 0x5, 0x2, 0x8, 0xc, 0x3, 0x7, 0x0, 0x4, 0xa, 0x1, 0xd, 0xb, 0x6},
    {0x4, 0x3, 0x2, 0xc, 0x9, 0x5, 0xf, 0xa, 0xb, 0xe, 0x1, 0x7, 0x6, 0x0, 0x8, 0xd}},
                           
//S7 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
   {{0x4, 0xb, 0x2, 0xe, 0xf, 0x0, 0x8, 0xd, 0x3, 0xc, 0x9, 0x7, 0x5, 0xa, 0x6, 0x1},
    {0xd, 0x0, 0xb, 0x7, 0x4, 0x9, 0x1, 0xa, 0xe, 0x3, 0x5, 0xc, 0x2, 0xf, 0x8, 0x6},
    {0x1, 0x4, 0xb, 0xd, 0xc, 0x3, 0x7, 0xe, 0xa, 0xf, 0x6, 0x8, 0x0, 0x5, 0x9, 0x2},
    {0x6, 0xb, 0xd, 0x8, 0x1, 0x4, 0xa, 0x7, 0x9, 0x5, 0x0, 0xf, 0xe, 0x2, 0x3, 0xc}},
                           
//S8 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
   {{0xd, 0x2, 0x8, 0x4, 0x6, 0xf, 0xb, 0x1, 0xa, 0x9, 0x3, 0xe, 0x5, 0x0, 0xc, 0x7},
    {0x1, 0xf, 0xd, 0x8, 0xa, 0x3, 0x7, 0x4, 0xc, 0x5, 0x6, 0xb, 0x0, 0xe, 0x9, 0x2},
    {0x7, 0xb, 0x4, 0x1, 0x9, 0xc, 0xe, 0x2, 0x0, 0x6, 0xa, 0xd, 0xf, 0x3, 0x5, 0x8},
    {0x2, 0x1, 0xe, 0x7, 0x4, 0xa, 0x8, 0xd, 0xf, 0xc, 0x9, 0x0, 0x3, 0x5, 0x6, 0xb}}
};

//These tables follow the bit orientation given in the official DES documentation.

//Initial permutation
static uint32_t ip_i = 64;
static uint32_t ip_o = 64;
static uint32_t ip[] = {    
    58, 50, 42, 34, 26, 18, 10, 2, 
    60, 52, 44, 36, 28, 20, 12, 4, 
    62, 54, 46, 38, 30, 22, 14, 6, 
    64, 56, 48, 40, 32, 24, 16, 8, 
    57, 49, 41, 33, 25, 17, 9, 1, 
    59, 51, 43, 35, 27, 19, 11, 3, 
    61, 53, 45, 37, 29, 21, 13, 5, 
    63, 55, 47, 39, 31, 23, 15, 7  };

//Initial key permutation
static uint32_t ikp_i = 64;
static uint32_t ikp_o = 56;
static uint32_t ikp[] = {
    57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4};
//round key permutation
static uint32_t rkp_i = 56;
static uint32_t rkp_o = 48;
static uint32_t rkp[] = {
    14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10,
    23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32  };
//Round block permutation
static uint32_t rp_i = 32;
static uint32_t rp_o = 32;
static uint32_t rp[] = {16, 7, 20, 21, 29, 12, 28, 17, 
        1, 15, 23, 26, 5, 18, 31, 10, 
        2, 8, 24, 14, 32, 27, 3, 9, 
        19, 13, 30, 6, 22, 11, 4, 25};
static inline uint64_t extractBit(uint64_t input, int src, int dst, int s_size, int d_size) {
    dst = (d_size - dst);
    src = (s_size - src);
    uint64_t bit = (uint64_t)0x1 << (src);
    uint64_t output;
    if (src >= dst) 
        output =  (input & bit) >> (src - dst);
    else
        output = (input & bit) << (dst - src);
    #ifdef DEBUG_EXTRACTOR
        printf("extractBit:\tsrc:%u\tdst:%u\t\tI:%16llX\tO:%16llX\n", src, dst, input, output); 
    #endif
    return output;
}

static uint64_t doPermute(uint64_t input, uint32_t* mapping, uint32_t s_size, uint32_t d_size) {
    uint64_t output = 0;
    for (int i = 1; i <= d_size; i++) {
        output |= extractBit(input, mapping[i - 1], i, s_size, d_size);
    }
    return output;
}

static uint64_t doInversePermute(uint64_t input, uint32_t* mapping, uint32_t s_size, uint32_t d_size) {
    uint64_t output = 0;
    for (int i = 1; i <= s_size; i++) {
        output |= extractBit(input, i, mapping[i - 1], s_size, d_size);
    }
    return output;
} 

static uint32_t doSBoxSub(int s_selector, uint32_t input) {
    uint32_t left = (input & 0x20)>>4; //Grab the outer bits
    uint32_t right = input & 0x1;
    uint32_t r_selector = left | right; //Row selector
    uint32_t c_selector = (input & 0x1E)>>1; //Grab middle 4 bits, column selector
    uint32_t toret = sbox[s_selector][r_selector][c_selector];
    #ifdef DEBUG_SBOX
        printf("doSBoxSub: input: %2x s_selector: %x r_selector: %x c_selector: %x  output: %x\n", input, s_selector, r_selector, c_selector, toret);
    #endif
    return toret;
}


static uint64_t initialPermutation(uint64_t input) {
    uint64_t output = doPermute(input, ip, ip_i, ip_o);
    #ifdef DEBUG_PERMUTER
        printf("initialPermutation: B:%16llx A:%16llx\n", input, output);
    #endif
    return output;
}

static uint64_t finalPermutation(uint64_t input) {
    uint64_t output = doInversePermute(input, ip, ip_i, ip_o);
    #ifdef DEBUG_PERMUTER
        printf("finalPermutation: B:%16llx A:%16llx\n", input, output);
    #endif
    return output;
}

static uint64_t generateCD(uint64_t input) {
    uint64_t output = doPermute(input, ikp, ikp_i, ikp_o);
    return output;
}

static uint64_t generatePerRoundKey(uint64_t CD) {
    uint64_t output = doPermute(CD, rkp, rkp_i, rkp_o);
    return output;
}

static uint64_t rotateCD(int round_num, uint64_t CD) {
   int rotation = 2;
    if (round_num == 0 || round_num == 1 || round_num == 8 || round_num == 15) 
       rotation = 1;
    uint32_t C = 0;
    uint32_t D = 0;
    uint64_t tmpC = CD >> 28;
    uint64_t tmpD = CD & 0xFFFFFFF;
    C = tmpC >> (28 - rotation);
    C = (C | ( tmpC << rotation)) & 0xFFFFFFF;
    D = tmpD >> (28 - rotation);
    D = (D | ( tmpD << rotation)) & 0xFFFFFFF;
    CD = ((uint64_t)C << 28) | D;
    #ifdef DEBUG_ROTATE
        printf("rotateRoundKey:\tnum:%u\tCB: %llX\tCA: %llX\tDB: %llX\tDA:%llX\n", round_num, tmpC, C, tmpD, D);
    #endif
    return CD;
}


static uint32_t manglerFunction(uint64_t key, uint32_t R) {
    uint32_t output = 0x0;
    uint32_t mask = 0x3F;
    uint32_t block_r;
    uint32_t block_k;
    #ifdef DEBUG_MANGLER
        uint64_t E = 0;
        uint64_t EKS = 0;
    #endif
    for (int i = 0; i < 8; i++ ) {
        if (i == 0) { 
            block_r = ((R & 0x1F) << 1);
        }
        else {
            block_r = (R & (mask << ((i * 4) - 1))) >> ((i * 4) - 1);
            
        }
        if (i == 0) {
            block_r = block_r | ((R & 0x80000000)>>31);
        }
        if (i == 7){
            block_r = block_r | ((R & 1) << 5);
        }
        block_k = (((uint64_t)0x3F << (i * 6)) & key) >> (i * 6);
        uint32_t block = block_r^block_k;
        #ifdef DEBUG_MANGLER
            E |= ((uint64_t)block_r << (i * 6));
            EKS |= ((uint64_t)block << (i * 6));
  
        #endif
        output |= ((doSBoxSub(7 - i, block)) << (i*4)); 
    }
    uint32_t perm = doPermute(output, rp, rp_i, rp_o);
    #ifdef DEBUG_MANGLER 
        printf("MANGLER: E: %14llx KS: %14llx E^KS: %14llx SBox: %8llx Perm: %8llx\n",E, key, EKS, output, perm);
    #endif
    return perm;
}

static uint64_t expandParity(uint64_t key) {
    uint64_t output = 0;
    static uint32_t mask = 0x7FFF;
    for (int i = 0; i < 8; i++) {
        output |= (key & (0x7F << i * 7)) <<((i * 8)-(i * 7));
    }
    printf("KEXP: %16llx\n", output);
    return output;
}

static uint64_t computeEncryptionRound(int round_num, uint64_t input, uint64_t key) {
    uint32_t R = input;
    uint32_t L = (uint64_t)(input >> 32);
    uint64_t output = (uint64_t)(R) << 32;
    uint32_t mangle = manglerFunction(key, R);
    output = output | mangle^L;
    #ifdef DEBUG_ROUNDS
    printf("f(R%u\t= %8x, L\t= %8x  SK%u\t= %3x %3x %3x %3x %3x %3x %3x %3x) = %llx\n", round_num, R, L, round_num + 1, key>>42,
                                                                                    key>>36 & 0x3F,
                                                                                    key>>30 & 0x3F,
                                                                                    key>>24 & 0x3F,
                                                                                    key>>18 & 0x3F,
                                                                                    key>>12 & 0x3F,
                                                                                    key>>6 & 0x3F,
                                                                                    key & 0x3F,
                                                                                    mangle);
    #endif
    return output;
}

static uint64_t computeDecryptionRound(int round_num, uint64_t input, uint64_t key) {
    uint32_t R = input;
    uint32_t L = (uint64_t)(input >> 32);
    uint64_t output = L;
    output = output | ((uint64_t)(manglerFunction(key, L)^R)<<32);
    return output;       
}

uint64_t DESEncrypt(uint64_t input, uint64_t key) {
    input = initialPermutation(input);
    #ifdef DEBUG_ROUNDS
        printf("IP: %16llx\n", input);
    #endif
    uint64_t keys[16];
    uint64_t CD = generateCD(key);
    #ifdef DEBUG_KEYGEN
        printf("CD:\t%16llx\n", CD);
    #endif
    for (int i = 0; i < 16; i++) {
        CD = rotateCD(i, CD);
        keys[i] = generatePerRoundKey(CD);
        #ifdef DEBUG_KEYGEN
        printf("K%u:\t%16llx\tCD:%16llx\n", i, keys[i], CD); 
        #endif
    }
    for (int round_num = 0; round_num < 16; round_num++) {
        input = computeEncryptionRound(round_num, input, keys[round_num]);
    }
    //Endianess issues require 32-bit block swap
    input = (input >> 32) | (input  << 32);
    uint64_t output = finalPermutation(input);
    #ifdef DEBUG_ROUNDS                                                          
        printf("F: %16llx FP: %16llx\n", input, output);   
    #endif
    return output; 
}

uint64_t DESDecrypt(uint64_t input, uint64_t key) {
    input = initialPermutation(input);
    input = (input >> 32) | (input  << 32);
    #ifdef DEBUG_ROUNDS
        printf("IP: %16llx\n", input);
    #endif
    uint64_t keys[16];
    uint64_t CD = generateCD(key);
    #ifdef DEBUG_KEYGEN
        printf("CD:\t%16llx\n", CD);
    #endif
    for (int i = 0; i < 16; i++) {
        CD = rotateCD(i, CD);
        keys[i] = generatePerRoundKey(CD);
        #ifdef DEBUG_KEYGEN
        printf("K%u:\t%16llx\tCD:%16llx\n", i, keys[i], CD); 
        #endif
    }
    for (int round_num = 15; round_num >= 0; round_num--) {
        input = computeDecryptionRound(round_num, input, keys[round_num]);
    }
    return finalPermutation(input);
}


typedef struct {
    uint32_t thread_id;
    uint64_t data;
    uint32_t size;
    uint64_t* array;
    uint64_t key_start;
    uint64_t* num_checked;
} BruteForceData;

static void* startBruteForceEncryptThread(void* arg) {
    BruteForceData* bf_data = (BruteForceData*)arg;
    fprintf(stderr, "Thread %3u started. Keys: %llX ->%llX\n", bf_data->thread_id,
                                                                bf_data->key_start,
                                                                bf_data->key_start + bf_data->size - 1);
    *(bf_data->num_checked) = 0;
    for (uint64_t key = bf_data->key_start, i = 0; i < bf_data->size; key++, i++) {
        bf_data->array[i] = DESEncrypt(bf_data->data, key);
        *(bf_data->num_checked)+=1;
    }
    return NULL; 
}

static void* startBruteForceDecryptThread(void* arg) {
    BruteForceData* bf_data = (BruteForceData*)arg;
    fprintf(stderr, "Thread %3u started. Keys: %llX ->%llX\n", bf_data->thread_id,
                                                                bf_data->key_start,
                                                                bf_data->key_start + bf_data->size - 1);
    *(bf_data->num_checked) = 0;
    for (uint64_t key = bf_data->key_start, i = 0; i < bf_data->size; key++, i++) {
        bf_data->array[i] = DESDecrypt(bf_data->data, key);
        *(bf_data->num_checked)+=1;
    }
    return NULL; 
}

typedef struct {
    uint64_t* num_checked;
    uint32_t num_threads;
    double start_time;
} StatsStruct;
/*
double get_time()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + t.tv_nsec * 1e-6;//t.tv_sec + t.tv_nsec*1e-6;
}
*/
double get_time()
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}

inline uint64_t getTotal(StatsStruct* stats) {
    uint64_t total = 0;
    for (int i = 0; i < stats->num_threads; i++)
        total += stats->num_checked[i];
    return total;
}

void printFinalStats(StatsStruct* stats) {
    float total_time = get_time() - stats->start_time;
    uint64_t total = getTotal(stats);
    printf("%llu keys checked in %f seconds at %f keys/s.\n", total, total_time, total/total_time);
}

void* printStatistics(void* arg) {
    StatsStruct* stats = (StatsStruct*)arg;
    double start_time = get_time();
    stats->start_time = start_time;
	double c_time;
    uint64_t total;
    for(;;) {
        sleep(1);
		c_time = get_time();
        total = getTotal(stats);
        printf("%llu = keys checked at ~ %f keys/s. t = %f s\n", total,(total)/((c_time - start_time)), c_time - start_time );
    }
} 

static int startBruteForce(void* (thread_ptr)(void*), 
                            uint64_t* output,
                            uint32_t size,
                            uint64_t key_start,
                            uint64_t data,
                            uint32_t num_threads) {
    pthread_t threads[num_threads];
    uint64_t num_checked[num_threads];
    BruteForceData bf_data[num_threads];
    int err;
    uint32_t division = size / num_threads;
    pthread_t stats_thread;
    StatsStruct stats;
    stats.num_checked = num_checked;
    stats.num_threads = num_threads;
    pthread_create(&stats_thread, NULL, printStatistics, (void*) &stats); 
    for (int i = 0; i < num_threads; i++) {
        bf_data[i].thread_id = i;
        bf_data[i].data = data;
        bf_data[i].size = division;
        bf_data[i].array = output + (division * i);
        bf_data[i].key_start = key_start + (division * i);
        bf_data[i].num_checked = &num_checked[i]; 
        err = pthread_create(&threads[i], NULL, thread_ptr, (void*) &bf_data[i]);
        if (err != 0) {
            //TODO error code handling
            fprintf(stderr, "Unable to create worker threads.\n");
            return 0;
        } 
    }
    for (int i = 0; i < num_threads;i++) {
        pthread_join(threads[i], NULL);
    }
    printFinalStats(&stats);
    pthread_cancel(stats_thread);
    return 1;
}
                            

int DESBruteForceEncrypt(uint64_t* output,                                      
                          uint32_t size,                                         
                          uint64_t key_start,
                          uint64_t data,                                     
                          uint32_t num_threads) {
    return startBruteForce(startBruteForceEncryptThread, output, size, key_start, data, num_threads);
}                            
                                                                                 
int DESBruteForceDecrypt(uint64_t* output,                                      
                          uint32_t size,                                         
                          uint64_t key_start,
                          uint64_t data,
                          uint32_t num_threads) {
    return startBruteForce(startBruteForceDecryptThread, output, size, key_start, data, num_threads);
}

#ifdef OPENCL

#define BLOCK_SIZE 1048576 

typedef struct {
    cl_platform_id platform;                                                     
    cl_context context;                                                          
    cl_command_queue queue;                                                      
    cl_device_id device;                                                         
    cl_program program;                                                          
    cl_kernel krn_enc;  
    cl_kernel krn_dec;  
    cl_mem output;
} OpenCLData;


void checkCLError(int error, const char* position) {
    if (error != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error Thrown at %s. %i\n", position, error);
        exit(1);   
    }
}

int initOpenCL(OpenCLData* data, int cl_size) {
    //We will initialize the opencl stuff here
    cl_int error = 0;
    // Platform
    error = clGetPlatformIDs(1, &(data->platform), NULL);
    checkCLError(error, "clGetPlatformIDs");
    // Device
    error = clGetDeviceIDs(data->platform, CL_DEVICE_TYPE_GPU, 1, 
                           &(data->device), NULL);
    checkCLError(error, "clGetDeviceIDs");
    // Context
    data->context = clCreateContext(NULL, 1, &(data->device), NULL, NULL, &error);
    checkCLError(error, "clCreateContext");
    // Command-queue
    data->queue = clCreateCommandQueue(data->context, data->device, 0, &error);
    checkCLError(error, "clCreateCommandQueue");



                                                                               
    const char* FILE_NAME = "des.cl";
    FILE *fp;                                                                    
    fp = fopen(FILE_NAME, "r");                                                  
    if (!fp) {                                                                   
        fprintf(stderr, "Failed to load the kernel.\n");                         
        exit(1);                                                                 
    }                                                                            
    fseek(fp, 0, SEEK_END);                                                      
    int sz = ftell(fp);                                                          
    fseek(fp, 0, SEEK_SET);                                                      
    char* buf = malloc((sz + 1) * sizeof(char));
	if (buf == NULL) {
		fprintf(stderr, "Out of memory.\n");
		exit(1);
	} 
    buf[sz] = '\0';                                                              
    fread(buf, sizeof(char), sz, fp);                                            
    fclose(fp);                                                                  
                                                                                 
    data->program = clCreateProgramWithSource(data->context, 1
                                              , (const char **)&buf, 
                                             NULL, &error);
    checkCLError(error, "clCreateProgramWithSource");                                                 
    error = clBuildProgram(data->program, 0, NULL, NULL, NULL, NULL); 
    if (error != CL_SUCCESS) { 
        char log[10000];                                                          
        error = clGetProgramBuildInfo(data->program, data->device, 
                                      CL_PROGRAM_BUILD_LOG, sizeof(char[10000]), 
                                      &log , NULL);
        printf("Log %i:\n%s\n",error, log);                                       
    }                                                                            
    checkCLError(error, "clGetProgramBuildInfo");                                                 
                                                                                 
    data->krn_enc = clCreateKernel(data->program, "des_encrypt_kern", &error);                       
    data->krn_dec = clCreateKernel(data->program, "des_decrypt_kern", &error); 
    checkCLError(error, "");                                                 
    free(buf);  
    return error;
}

int DESBruteForceCL(      OpenCLData* cl_data,
                          cl_kernel krn,
                          uint64_t* output, 
                          uint32_t size,                                         
                          uint64_t key_start,                                    
                          uint64_t data) {

    pthread_t stats_thread;
    StatsStruct stats;
    uint64_t total = 0;
    stats.num_checked = &total;
    stats.num_threads = 1;
    pthread_create(&stats_thread, NULL, printStatistics, (void*) &stats);
    size_t blocks;
    size_t final_block;
    blocks = (size / BLOCK_SIZE) + 1;
    final_block = size % BLOCK_SIZE;
    cl_int error = 0;                                                           
    cl_data->output = clCreateBuffer(cl_data->context, CL_MEM_WRITE_ONLY, 
                                  min(BLOCK_SIZE, size) * sizeof(cl_ulong), NULL, &error);
    checkCLError(error, "clCreateBuffer"); 
    size = BLOCK_SIZE;
    for (int i = 0; i < blocks; i++) {
        if (i == blocks -1)
            size = final_block;
        size_t global_ws = size / 4;
        error  = clSetKernelArg(krn, 0, sizeof(cl_ulong), &key_start);
        error |= clSetKernelArg(krn, 1, sizeof(cl_ulong), &data);              
        error |= clSetKernelArg(krn, 2, sizeof(cl_mem), &(cl_data->output));                    
        checkCLError(error, "clSetKernelArg");                                         
                                                                                 
        cl_event wait;                                                       
        error = clEnqueueNDRangeKernel(cl_data->queue, krn, 1, NULL, &global_ws, NULL, 0, NULL, &wait); 
	    checkCLError(error, "clEnqueueNDRangeKernel");                                         
                                                                                 
        error = clWaitForEvents(1, &wait);                                   
        checkCLError(error, "clWaitForEvents");                                         
                                                                                 
        error = clEnqueueReadBuffer(cl_data->queue, cl_data->output, CL_TRUE, 0, 
									size * sizeof(cl_ulong), 
                                    output + BLOCK_SIZE * i, 
									0, NULL, NULL); 
        checkCLError(error, "clEnqueueReadBuffer");
        key_start += BLOCK_SIZE;
        total += size; 
    }
    printFinalStats(&stats);
    pthread_cancel(stats_thread);
    return 1; 
}

inline int testSize(uint32_t size) {
    uint32_t newsize = (size / 1024) * 1024;
    if (newsize != size) {
        fprintf(stderr, "Warning: OpenCL search size must be a multiple of 1024.\n");
        fprintf(stderr, "         Using %u.\n", newsize);
        size = newsize;
    }
    return (size);
}

                                                                          
int DESBruteForceEncryptCL(uint64_t* output,                                    
                          uint32_t size,                                         
                          uint64_t key_start,                                    
                          uint64_t tdata) {
    size = testSize(size);
    OpenCLData data;
    int error;
    if ( error = initOpenCL(&data, size) != CL_SUCCESS) {
        fprintf(stderr, "Unabled to open OpenCL Context. Error Code: %x\n", error);
        return 0; 
    }
    return DESBruteForceCL(&data, data.krn_enc, output, size, key_start, tdata);
}

int DESBruteForceDecryptCL(uint64_t* output,                                    
                          uint32_t size,                                         
                          uint64_t key_start,                                    
                          uint64_t tdata) {
    size = testSize(size);
    OpenCLData data;
    int error;
    if (error = initOpenCL(&data, size) != CL_SUCCESS) {
        fprintf(stderr, "Unabled to open OpenCL Context. Error Code: %x\n", error);
        return 0; 
    }
    return DESBruteForceCL(&data, data.krn_dec, output, size, key_start, tdata);
}

#else
int DESBruteForceEncryptCL(uint64_t* output,                                    
                          uint32_t size,                                         
                          uint64_t key_start,                                    
                          uint64_t data) {
    fprintf(stderr, "Unabled to open OpenCL Context.\n OpenCL Support was not enabled at compile time\n");
    return 0; 
}

int DESBruteForceDecryptCL(uint64_t* output,                                    
                          uint32_t size,                                         
                          uint64_t key_start,                                    
                          uint64_t data) {
    fprintf(stderr, "Unabled to open OpenCL Context.\n OpenCL Support was not enabled at compile time\n");
    return 0; 
}




#endif

static int test(char* msg, int assertion) {
    printf(msg);
    for (int i = strlen(msg); i < 72; i++)
        putchar(' ');
    if (assertion)
        printf("\e[1;32mPASSED\e[00m\n");
    else
        printf("\e[1;31mFAILED\e[00m\n");
    return assertion;
}

inline static uint64_t getRand64() {
    return (((uint64_t)rand() << 32) + rand()) ^ (((uint64_t)rand() << 32) + rand());

}

#ifdef UNIT_TESTS
int DESTestCases() {
    //Test cases
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    test("Testing single bit extraction:", extractBit(mask, 50, 64, 64, 64) == 0x1);
    
    uint64_t data = 0xFEEDACABDEADBEEF;
    uint64_t permute = initialPermutation(data);
    test("Testing initial and final permutations:", data == finalPermutation(permute));

        
    data = 0x675A69675E5A6B5A;
    uint64_t key = 0x5B5A57676A56676E;
    uint64_t cipherdata = DESEncrypt(data, key);
    uint64_t newdata = DESDecrypt(cipherdata, key);
//    printf("ACT: K: %16llX\t\tD: %16llX\t\t C: %16llX\t\t ND: %16llX\t\t\n", key, data, cipherdata, newdata);
//    printf("REF: K: 5B5A57676A56676E\t\tD: 675A69675E5A6B5A\t\t C: 974AFFBF86022D1F\t\t ND: 675A69675E5A6B5A\n");
    test("Testing encryptiong inverse round:", data == newdata);
    test("Testing correct output of encryption (PASS 1):", cipherdata == 0x974AFFBF86022D1F);
    test("Testing correct output of encryption (PASS 2):", DESEncrypt(0xAAAAAAAAAAAAAAAA, 0x3b3898371520f75e) == 0xfec0f6eaffd979f8);
    int good = 1;
    srand(time(NULL));
    for (int i = 0; i < 100000; i++) {
        key = getRand64();
        data = getRand64();
        newdata = DESDecrypt(DESEncrypt(data, key), key);
        good = (newdata==data); 
    }
    test("Testing 100000 random encryption rounds:", good);

    #ifdef OPENCL
    const uint32_t ARRAY_SIZE = 4096;
    data = 0x675A69675E5A6B5A;
    key = 0x5B5A57676A56676E;
    uint64_t test_array_cl[ARRAY_SIZE];
    uint64_t test_array[ARRAY_SIZE];
    DESBruteForceEncryptCL(test_array_cl, ARRAY_SIZE, key, data);
    DESBruteForceEncrypt(test_array, ARRAY_SIZE, key, data, 1);
    good = 1;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        //printf("CL:%16llX C:%16llX\n", test_array_cl[i], test_array[i]);
        if (test_array[i] != test_array_cl[i]) {
            fprintf(stderr,"Descrepancy Offset: %u, WAS:%16llX EXPECTED:%16llX\n",i, test_array_cl[i], test_array[i]);
            good = 0;
            break;
        }
    }
    test("Testing OpenCL Encryption", good);
    good = 1;
    DESBruteForceDecryptCL(test_array_cl, ARRAY_SIZE, key, data);
    DESBruteForceDecrypt(test_array, ARRAY_SIZE, key, data, 1);
    good = 1;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        //printf("CL:%16llX C:%16llX\n", test_array_cl[i], test_array[i]);
        if (test_array[i] != test_array_cl[i]) {
            fprintf(stderr,"Descrepancy Offset: %u, WAS:%16llX EXPECTED:%16llX\n",i, test_array_cl[i], test_array[i]);
            good = 0;
            break;
        }
    }
    test("Testing OpenCL Decryption", good);
    return !good; 
    #endif
}
#endif
