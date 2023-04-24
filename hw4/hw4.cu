//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch
// and solving a block (#286819) which the information is downloaded from Block Explorer
//***********************************************************************************

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "sha256.h"

////////////////////////   Block   /////////////////////

#define BLOCK 8192
#define THREAD 64

__constant__ const WORD loopLimit = 0xffffffff / (THREAD * BLOCK);

typedef struct _block {
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
} HashBlock;

__device__ void copyBlock(HashBlock *target, HashBlock *source) {
    target->version = source->version;
    target->ntime = source->ntime;
    target->nbits = source->nbits;
    target->nonce = source->nonce;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        target->prevhash[i] = source->prevhash[i];
        target->merkle_root[i] = source->merkle_root[i];
    }
}
__device__ void copySHA256(SHA256 *target, SHA256 *source) {
#pragma unroll
    for (int i = 0; i < 8; i++)
        target->h[i] = source->h[i];
#pragma unroll
    for (int i = 0; i < 32; i++)
        target->b[i] = source->b[i];
}

////////////////////////   Utils   ///////////////////////

// convert one hex-codec char to binary
unsigned char decode(unsigned char c) {
    switch (c) {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c - '0';
    }
}

// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char *out, char *in, size_t string_len) {
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len / 2 - 1;

    for (s, b; s < string_len; s += 2, --b) {
        out[b] = (unsigned char)(decode(in[s]) << 4) + decode(in[s + 1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char *hex, size_t len) {
    for (int i = 0; i < len; ++i) {
        printf("%02x", hex[i]);
    }
}

// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char *hex, size_t len) {
    for (int i = len - 1; i >= 0; --i) {
        printf("%02x", hex[i]);
    }
}
__device__ void print_hex_inverse_device(unsigned char *hex, size_t len) {
    for (int i = len - 1; i >= 0; --i) {
        printf("%02x", hex[i]);
    }
}

__device__ int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len) {
    // compared from lowest bit

    for (int i = byte_len - 1; i >= 0; --i) {
        if (a[i] < b[i])
            return -1;
        else if (a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp) {
    int i = 0;
    while (i < len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n')
        ;
    str[len - 1] = '\0';
}

////////////////////////   Hash   ///////////////////////

__device__ void double_sha256_device(SHA256 *sha256_ctx, unsigned char *bytes) {
    SHA256 tmp;
    sha256_device(&tmp, (BYTE *)bytes, sizeof(HashBlock));
    sha256_device(sha256_ctx, (BYTE *)&tmp, sizeof(SHA256));
}

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len) {
    SHA256 tmp;
    sha256(&tmp, (BYTE *)bytes, len);
    sha256(sha256_ctx, (BYTE *)&tmp, sizeof(tmp));
}

////////////////////   Merkle Root   /////////////////////

// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch) {
    size_t total_count = count;  // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count + 1) * 32];
    unsigned char **list = new unsigned char *[total_count + 1];

    // copy each branch to the list
    for (int i = 0; i < total_count; ++i) {
        list[i] = raw_list + i * 32;
        // convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count * 32;

    // calculate merkle root
    while (total_count > 1) {
        // hash each pair
        int i, j;

        if (total_count % 2 == 1) {  // odd,
            memcpy(list[total_count], list[total_count - 1], 32);
        }

        for (i = 0, j = 0; i < total_count; i += 2, ++j) {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256 *)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}

__global__ void solveLoop(HashBlock *blockDevice, unsigned char *target_hex, bool *isEnd) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ HashBlock block[THREAD];
    __shared__ SHA256 sha256_ctx[THREAD];
    copyBlock(&block[threadIdx.x], blockDevice);
    block[threadIdx.x].nonce = idx * loopLimit;
    for (WORD t = 0; t <= loopLimit; t++) {
        // sha256d
        if (*isEnd)
            break;
        double_sha256_device(&sha256_ctx[threadIdx.x], (unsigned char *)&block[threadIdx.x]);
        /*if (block[threadIdx.x].nonce % 1000000 == 0) {
            printf("hash #%10u (big): ", block[threadIdx.x].nonce);
            print_hex_inverse_device(sha256_ctx.b, 32);
            printf("\n");
        }*/

        if (little_endian_bit_comparison(sha256_ctx[threadIdx.x].b, target_hex, 32) < 0)  // sha256_ctx < target_hex
        {
            // printf("Found Solution!!\n");
            // printf("hash #%10u (big): ", block[threadIdx.x].nonce);
            // print_hex_inverse_device(sha256_ctx[threadIdx.x].b, 32);
            // printf("\n\n");
            // copySHA256(sha256_ctxDevice, &sha256_ctx[threadIdx.x]);
            copyBlock(blockDevice, &block[threadIdx.x]);
            *isEnd = true;
            break;
        }

        block[threadIdx.x].nonce++;
    }
}
void solve(FILE *fin, FILE *fout, int totalblock) {
    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    HashBlock *blockDevice;
    SHA256 *sha256_ctxDevice;
    unsigned char *target_hexDevice;
    bool *isEndDevice;
    cudaMalloc(&blockDevice, sizeof(HashBlock));
    cudaMalloc(&sha256_ctxDevice, sizeof(SHA256));
    cudaMalloc(&target_hexDevice, sizeof(unsigned char) * 32);
    cudaMalloc(&isEndDevice, sizeof(bool));

    for (int i = 0; i < totalblock; ++i) {
        getline(version, 9, fin);
        getline(prevhash, 65, fin);
        getline(ntime, 9, fin);
        getline(nbits, 9, fin);
        fscanf(fin, "%d\n", &tx);
        // printf("%d\n",tx);

        raw_merkle_branch = new char[tx * 65];
        merkle_branch = new char *[tx];
        for (int i = 0; i < tx; ++i) {
            merkle_branch[i] = raw_merkle_branch + i * 65;
            getline(merkle_branch[i], 65, fin);
            merkle_branch[i][64] = '\0';
        }

        // **** calculate merkle root ****

        unsigned char merkle_root[32];
        calc_merkle_root(merkle_root, tx, merkle_branch);

        HashBlock block;

        // convert to byte array in little-endian
        convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
        convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
        memcpy(block.merkle_root, merkle_root, 32);
        convert_string_to_little_endian_bytes((unsigned char *)&block.nbits, nbits, 8);
        convert_string_to_little_endian_bytes((unsigned char *)&block.ntime, ntime, 8);
        block.nonce = 0;

        // ********** calculate target value *********
        // calculate target value from encoded difficulty which is encoded on "nbits"
        unsigned int exp = block.nbits >> 24;
        unsigned int mant = block.nbits & 0xffffff;
        unsigned char target_hex[32] = {};

        unsigned int shift = 8 * (exp - 3);
        unsigned int sb = shift / 8;
        unsigned int rb = shift % 8;

        // little-endian
        target_hex[sb] = (mant << rb);
        target_hex[sb + 1] = (mant >> (8 - rb));
        target_hex[sb + 2] = (mant >> (16 - rb));
        target_hex[sb + 3] = (mant >> (24 - rb));


        // ********** find nonce **************

        //SHA256 sha256_ctx;

        bool isEnd = false;

        cudaMemcpy(blockDevice, &block, sizeof(HashBlock), cudaMemcpyHostToDevice);
        // cudaMemcpy(sha256_ctxDevice, &sha256_ctx, sizeof(SHA256), cudaMemcpyHostToDevice);
        cudaMemcpy(target_hexDevice, &target_hex, sizeof(unsigned char) * 32, cudaMemcpyHostToDevice);
        cudaMemcpy(isEndDevice, &isEnd, sizeof(bool), cudaMemcpyHostToDevice);

        solveLoop<<<BLOCK, THREAD>>>(blockDevice, target_hexDevice, isEndDevice);
        cudaDeviceSynchronize();
        //cudaMemcpy(&sha256_ctx, sha256_ctxDevice, sizeof(SHA256), cudaMemcpyDeviceToHost);
        cudaMemcpy(&block, blockDevice, sizeof(HashBlock), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < 4; ++i) {
            fprintf(fout, "%02x", ((unsigned char *)&block.nonce)[i]);
        }
        fprintf(fout, "\n");

        
        delete[] merkle_branch;
        delete[] raw_merkle_branch;
    }
    cudaDeviceReset();
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    solve(fin, fout, totalblock);

    return 0;
}
