#ifndef CUDA_METHODS_H
#define CUDA_METHODS_H

extern "C" void rgb2yuv(int *imgr,int *imgg,int *imgb,int *imgy,int *imgcb,int *imgcr, int n);
extern "C" void yuv2rgb(int *imgy,int *imgcb,int *imgcr, int *imgr,int *imgg,int *imgb, int n);
extern "C" void rgb2gray(unsigned char *imgr, unsigned char *imgg, unsigned char *imgb, unsigned char *img_gray, int n);
extern "C" void rgb2binary(unsigned char *imgr, unsigned char *imgg, unsigned char *imgb, unsigned char *img_binary, int n, int umbral);
extern "C" void histogram256(unsigned int *d_Histogram, void *d_Data, unsigned int byteCount);
extern "C" void setFogKernel(float h_Kernel[]);
extern "C" void fogFilter(unsigned char* src_h, unsigned char* dst_h,
                                unsigned int width, unsigned int height,
                                unsigned int pitch, float scale);
extern "C" void setSobelKernel(int hgx_Kernel[], int hgy_Kernel[]);
extern "C" void sobelFilter(unsigned char *src_h, unsigned char *dst_h,
                            unsigned int width, unsigned int height,
                            unsigned int pitch, float scale);
extern "C" void img2fft(unsigned char *src, unsigned char *dst, int w, int h);
extern "C" void addImage(unsigned char *imgr, unsigned char *imgg, unsigned char *imgb,
                         unsigned char *imgr_k, unsigned char *imgg_k, unsigned char *imgb_k,
                         int w, int h, float index);
extern "C" void imageScaled(unsigned char *src, int w, int h, int pitch, unsigned char *dst, int nw, int nh, int npitch, int BytesPerPixel);
extern "C" void GetMatch(float *I, float *T, int Iw, int Ih, int Tw, int Th, int *x, int *y);

#endif // CUDA_METHODS_H
