#include <omp.h>
#include <png.h>
#include <stdio.h>
#include <zlib.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

#define BLOCK 4096
#define THREAD 256
#define GRID 16
std::atomic_int passedH;
// clang-format off
__constant__ int mask[MASK_N][MASK_X][MASK_Y] = {
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0},
     {  2,  8, 12,  8,  2},
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1},
     { -4, -8,  0,  8,  4},
     { -6,-12,  0, 12,  6},
     { -4, -8,  0,  8,  4},
     { -1, -2,  0,  2,  1}}
};
// clang-format on

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
             unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }
#pragma omp simd
    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
#pragma omp simd
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("threadId %d %d %d\nblockId %d %d %d\nblockDim %d %d %d\ngridDim %d %d %d\nID %d\n\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, idx);
    // printf("%d\n", idx);
    //__shared__ unsigned char sharedImg[7 * 2 * THREAD * 3];
    int j = 0;

    j = 0;
    // for (int i = 0; i < 3; i++) {
    /*sharedImg[channels * (width * j + threadIdx.x + i) + 2] = s[channels * (width * ((idx / width - 3 < 0) ? 0 : (idx / width - 3)) + threadIdx.x - (3 - i)) + 2];
    sharedImg[channels * (width * j + threadIdx.x + i) + 1] = s[channels * (width * ((idx / width - 3 < 0) ? 0 : (idx / width - 3)) + threadIdx.x - (3 - i)) + 1];
    sharedImg[channels * (width * j + threadIdx.x + i)] = s[channels * (width * ((idx / width - 3 < 0) ? 0 : (idx / width - 3)) + threadIdx.x - (3 - i))];
    sharedImg[channels * (width * j + threadIdx.x + 3 + THREAD + i) + 2] = s[channels * (width * ((idx / width - 3 < 0) ? 0 : (idx / width - 3)) + threadIdx.x + 3 + THREAD + i) + 2];
    sharedImg[channels * (width * j + threadIdx.x + 3 + THREAD + i) + 1] = s[channels * (width * ((idx / width - 3 < 0) ? 0 : (idx / width - 3)) + threadIdx.x + 3 + THREAD + i) + 1];
    sharedImg[channels * (width * j + threadIdx.x + 3 + THREAD + i)] = s[channels * (width * ((idx / width - 3 < 0) ? 0 : (idx / width - 3)) + threadIdx.x + 3 + THREAD + i)];
    j++;*/
    //}
    /*if (threadIdx.x == 1) {
        for (int i = 0; i < 7 * (THREAD + 6) * 3; i++)
            printf("%d ", sharedImg[i]);
        printf("\n");
    }*/

    //__syncthreads();
    int x, y, i, v, u;
    int R, G, B;
    float val[MASK_N * 3] = {0.0};
    int adjustX, adjustY, xBound, yBound;

    // float maxColor = 255.0;
    //  int x = threadIdx.x + blockIdx.x * blockDim.x;
    //   int y = threadIdx.y + blockIdx.y * blockDim.y;
    //  printf("%d %d\n",cudax,cuday);
    adjustX = (MASK_X % 2) ? 1 : 0;
    adjustY = (MASK_Y % 2) ? 1 : 0;
    xBound = MASK_X / 2;
    yBound = MASK_Y / 2;
    for (j = idx; j < height * width; j += BLOCK * THREAD) {
        /*int startH = ((idx / width - 3 < 0) ? 0 : (idx / width - 3));
        for (int jj = startH; jj < ((idx / width + 3 < height) ? (idx / width - 3) : height); jj++) {
            for (int c = 0; c < 2; c++) {
                if (channels * (width * jj + j + c - 3) / 2 >= 0) {
                    sharedImg[channels * (width * (jj - startH) + threadIdx.x + c) + 2] = s[channels * (width * jj + j + c - 3) + 2];
                    sharedImg[channels * (width * (jj - startH) + threadIdx.x + c) + 1] = s[channels * (width * jj + j + c - 3) + 1];
                    sharedImg[channels * (width * (jj - startH) + threadIdx.x + c)] = s[channels * (width * jj + j + c - 3) / 2];
                }
            }
            printf("C\n");
            // sharedImg[channels * (width * j + threadIdx.x + 3) + 1] = s[channels * (width * i + threadIdx.x) + 1];
            // sharedImg[channels * (width * j + threadIdx.x + 3)] = s[channels * (width * i + threadIdx.x)];
        }
        __syncthreads();*/
        x = j % width;
        y = j / width;
        for (i = 0; i < MASK_N; ++i) {
            val[i * 3 + 2] = 0.0;
            val[i * 3 + 1] = 0.0;
            val[i * 3] = 0.0;
            for (v = -yBound; v < yBound + adjustY; ++v) {
                for (u = -xBound; u < xBound + adjustX; ++u) {
                    if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                        R = s[channels * (width * (y + v) + (x + u)) + 2];
                        G = s[channels * (width * (y + v) + (x + u)) + 1];
                        B = s[channels * (width * (y + v) + (x + u)) + 0];
                        // R = sharedImg[channels * ((2 * THREAD) * (3 + v) + (threadIdx.x + u + 3)) + 2];
                        // G = sharedImg[channels * ((2 * THREAD) * (3 + v) + (threadIdx.x + u + 3)) + 1];
                        // B = sharedImg[channels * ((2 * THREAD) * (3 + v) + (threadIdx.x + u + 3))];
                        val[i * 3 + 2] += R * mask[i][u + xBound][v + yBound];
                        val[i * 3 + 1] += G * mask[i][u + xBound][v + yBound];
                        val[i * 3 + 0] += B * mask[i][u + xBound][v + yBound];
                    }
                }
            }
        }

        float totalR = 0.0;
        float totalG = 0.0;
        float totalB = 0.0;
        for (i = 0; i < MASK_N; ++i) {
            totalR += val[i * 3 + 2] * val[i * 3 + 2];
            totalG += val[i * 3 + 1] * val[i * 3 + 1];
            totalB += val[i * 3 + 0] * val[i * 3 + 0];
        }
        t[channels * (width * y + x) + 2] = (totalR > 4161600.0) ? 255 : sqrt(totalR) / SCALE;
        t[channels * (width * y + x) + 1] = (totalG > 4161600.0) ? 255 : sqrt(totalG) / SCALE;
        t[channels * (width * y + x) + 0] = (totalB > 4161600.0) ? 255 : sqrt(totalB) / SCALE;
    }
}

int main(int argc, char** argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char* src_img = NULL;
    passedH = 0;
    auto start = std::chrono::steady_clock::now();
    read_png(argv[1], &src_img, &height, &width, &channels);
    auto end = std::chrono::steady_clock::now();
    assert(channels == 3);
    const unsigned imgSize = height * width * channels * sizeof(unsigned char);
    unsigned char* dst_img = (unsigned char*)malloc(imgSize);
    unsigned char* src_imgCuda;
    unsigned char* dst_imgCuda;
    auto IOTime = end - start;
    start = std::chrono::steady_clock::now();
    cudaMalloc(&src_imgCuda, imgSize);
    cudaMemcpy(src_imgCuda, src_img, imgSize, cudaMemcpyHostToDevice);
    cudaMalloc(&dst_imgCuda, imgSize);
    // cudaMemcpy(dst_imgCuda, dst_img, imgSize, cudaMemcpyHostToDevice);
    end = std::chrono::steady_clock::now();
    auto MemTime = end - start;
    
    start = std::chrono::steady_clock::now();
    // dim3 threads(32, 16, 1);
    // dim3 blocks(width / 32 + 1, height / 16 + 1, 1);
    sobel<<<BLOCK, THREAD>>>(src_imgCuda, dst_imgCuda, height, width, channels);

    // sobel<<<1, THREAD>>>(src_imgCuda, dst_imgCuda, height, width, channels, i);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    auto KernelTime = end - start;
    start = std::chrono::steady_clock::now();
    cudaMemcpy(dst_img, dst_imgCuda, imgSize, cudaMemcpyDeviceToHost);
    end = std::chrono::steady_clock::now();
    MemTime += (end - start);
    start = std::chrono::steady_clock::now();
    write_png(argv[2], dst_img, height, width, channels);
    end = std::chrono::steady_clock::now();
    IOTime += (end - start);
    // free memory
    cudaDeviceReset();
    free(src_img);
    free(dst_img);
    std::cout << "IO time: " << std::chrono::duration_cast<std::chrono::microseconds>(IOTime).count() << '\n';
    std::cout << "Memory time: " << std::chrono::duration_cast<std::chrono::microseconds>(MemTime).count() << '\n';
    std::cout << "Kernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(KernelTime).count() << '\n';
    return 0;
}
