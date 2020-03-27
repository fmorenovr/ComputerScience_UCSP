# include <cuda.h>
# include <cuda_runtime.h>

extern "C"
unsigned char * RGB2HSV(unsigned char * data, int npixels);

__global__ void RGB2HSVcuda(unsigned char * dataRGBdev, unsigned char * dataHSVdev, int npixels){
    int posThread = blockIdx.x*blockDim.x + threadIdx.x;
    // ** Size, just consider the number of pixel, non the total of data,
    // ** But in dataRGBdev is all channels data
    if(posThread < npixels){
        // Get the maximun & minimun value of RGB pixel
        unsigned char max = 0;
        unsigned char min = 255;
        int position = posThread*3;
        // Max pos represents 0: R, 1: G, 2: B
        int maxpos = 0;
        for(int i = 0; i < 3; i++){
            int pos = position + i;
            if(dataRGBdev[pos] > max){
                max = dataRGBdev[pos];
                maxpos = i;
            }
            else if(dataRGBdev[pos] < min)
                min = dataRGBdev[pos];
        }
        int diff = (max - min > 0)? max - min: 1;
        //set H position
        int valH;
        if(maxpos == 0)
            valH = ((dataRGBdev[position + 1] - dataRGBdev[position + 2])/diff) % 6;
        else if(maxpos == 1)
            valH = (dataRGBdev[position +2] - dataRGBdev[position])/diff + 2;
        else
            valH = (dataRGBdev[position] - dataRGBdev[position + 1])/diff +4;

        valH *=60;
        valH = (valH > 0) ? valH:0;
        valH = (valH < 255) ? valH:255;
        dataHSVdev[position] = (unsigned char) valH;
        // Set S position
        dataHSVdev[position +1 ] = (max == 0)? 0 : (max-min)/max;
        // Set V Position
        dataHSVdev[position + 2] = max;
    }
}

unsigned char * RGB2HSV(unsigned char * data, int npixels){
    unsigned char * hsv = new unsigned char[npixels*3];
    unsigned char * hsvDev;
    unsigned char * dataDev;
    cudaMalloc((void**)&hsvDev, npixels*3*sizeof(unsigned char));
    cudaMalloc((void**)&dataDev, npixels*3*sizeof(unsigned char));

    cudaMemcpy(dataDev, data, 3*npixels*sizeof(unsigned char), cudaMemcpyHostToDevice);
    int nThreads = 1024;
    int nBlocks = (npixels % nThreads > 0) ? npixels/nThreads + 1: npixels/nThreads;
    RGB2HSVcuda<<<nBlocks, nThreads>>>(dataDev, hsvDev, npixels); 
    cudaMemcpy(hsv, hsvDev, 3*npixels*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(dataDev);
    cudaFree(hsvDev);
    return hsv;
}
