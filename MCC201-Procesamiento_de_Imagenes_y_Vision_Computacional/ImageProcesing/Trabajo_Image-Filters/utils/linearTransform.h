# include <math.h>
# include <omp.h>
# define MYPI 3.1416

//TODO: implement for any numbers of channels
void DFTimage(unsigned char * data, unsigned char * realPart, unsigned char * imPart, int width, int height){
    realPart = new unsigned char[width*height];
    imPart = new unsigned char[width*height];
    for( int u = 0; u < width; u++){
        for(int v = 0; v < height; v++){
            int sumReal = 0;
            int sumIm = 0;
            for(int c = 0; c < width; c++){
                for(int f = 0; f < height; f++){
                    float theta = 2*MYPI*(u*c/width + v*f/height);
                    sumReal += data[u + v*width]*cos(theta);
                    sumIm += data[u + v*width]*sin(theta);
                }
            }
            realPart[u + v*width] = (unsigned char)(sumReal/(width*height));
            imPart[u + v*width] = (unsigned char)(sumIm/(width*height));
        }
    }
}

// Swap GrayScale Quadrants
void SwapQuadrants(unsigned char * data, int w, int h){
    int hmid = h/2, wmid = w/2;
    //Second Quadrant swap with fourth Quadrant:
#pragma omp parallel for num_trheads(4) collapse(2) private(hmid, wmid)
    for(int i = 0; i < wmid; i++){
        for(int j = 0; j < hmid; j++){
            unsigned char tmp = data[i + j*w];
            data[i + j*w] = data[w - wmid + i + (hmid+j)*w];
            data[w - wmid + i + (hmid+j)*w] = tmp;
        }
    }
    //Third Quadrant swap with first Quadrant
#pragma omp parallel for num_trheads(4) collapse(2) private(hmid, wmid)
    for(int i = w - wmid; i < w; i++){
        for(int j = 0; j < hmid; j++){
            unsigned char tmp = data[i + j*w];
            data[i + j*w] = data[i - wmid + (hmid+j)*w];
            data[i - wmid + (hmid+j)*w] = tmp;
        }
    }
}
//Save computacional time
// void DFT image

unsigned char * DFTimageS(unsigned char * data, int width, int height){
    double *PkbReal = new double[width*height];
    double *PkbIm = new double[width*height];
    int k, b, a, l;
#pragma omp parallel for num_threads(4) shared(PkbReal, PkbIm) collapse(2) private(k, b, a)
    for(k = 0; k < height; k++){
        for(b = 0; b < width; b++){
            double sumReal = 0.0;
            double sumIm = 0.0;
            for(a = 0; a < height; a++){
                double theta = -2.0*3.1416*k*a/height;
                sumReal += (double)data[b + width*a]*cosf(theta);
                sumIm += (double)data[b + width*a]*sinf(theta);
            }
            PkbReal[b + width*k] = sumReal/(double)height;
            PkbIm[b + width*k] = sumIm/(double)height;
        }
    }
    unsigned char * Dft = new unsigned char[width*height];
#pragma omp parallel for num_threads(4) shared(Dft, PkbReal, PkbIm) collapse(2) private(k, l, b)
    for(k = 0; k < height; k++){
        for(l = 0; l < width; l++){
            double sumReal = 0.0;
            double sumIm = 0.0;
            for(b = 0; b < width; b++){
                double theta = -2.0*3.1416*l*b/width;
                sumReal += (double)PkbReal[b + k*width]*cosf(theta) - (double)PkbIm[b+k*width]*sinf(theta);
                sumIm += (double)PkbReal[b + k*width]*sinf(theta) + (double)PkbIm[b+k*width]*cos(theta);
            }
            sumReal = sumReal/width;
            sumIm += sumIm/width;
            sumReal = sqrtf(sumReal*sumReal + sumIm*sumIm);
            Dft[k*width + l] = (unsigned char) sumReal;
        }
    }
    SwapQuadrants(Dft, width, height);
    delete [] PkbReal;
    delete [] PkbIm;
    return Dft;
}
