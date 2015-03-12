#include <stdio.h>
#include<cuda_runtime.h>
#include <time.h>
#include <cuda.h>

// CUDA runtime


// Helper functions and utilities to work with CUDA


#define N 256
//#define M 256


//__global__�����ĺ��������߱�������δ��뽻��CPU���ã���GPUִ��
__global__ void matrix_mult(float *dev_a, float* dev_b, float* dev_c, int Width)
{
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < Width) && (Col < Width)) {
	float Pvalue = 0;
	for (int k = 0; k < Width; k++)
	{
		Pvalue += dev_a[Row*Width + k] * dev_b[k*Width+Col];
	}
	dev_c[Row*Width + Col] = Pvalue;

}
}

int main(void)
{
	//���������ڴ棬�����г�ʼ��
	//clock_t start = clock();
	float host_a[N][N];
	float host_b[N][N];
	float host_c[N][N];
	for (int i = 0; i<N; i++)
	    for (int j = 0; j<N; j++)
	     host_a[i][j] = 1.0f;
	for (int i = 0; i<N; i++)
		for (int j = 0; j<N; j++)
		host_b[i][j] = 0.01f;
		/*
	for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
		printf("%d\n", host_a[i][j]);
	*/

	//����cudaError��Ĭ��ΪcudaSuccess(0)
	cudaError_t err = cudaSuccess;

	//����GPU�洢�ռ�
	float *dev_a, *dev_b, *dev_c;
	err = cudaMalloc((void **)&dev_a, sizeof(float)* N*N);
	err = cudaMalloc((void **)&dev_b, sizeof(float)* N*N);
	err = cudaMalloc((void **)&dev_c, sizeof(float)* N*N);
	if (err != cudaSuccess)
	{
		printf("the cudaMalloc on GPU is failed");
		return 1;
	}
	printf("SUCCESS");
	//��Ҫ���������ʹ��cudaMemcpy���͵�GPU
	cudaMemcpy(dev_a, host_a, sizeof(float)* N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, sizeof(float)* N*N, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice);
	
	//���ú˺�����GPU��ִ�С����ݽ��٣�֮ʹ��һ��Block������1024���߳�
    #define BLOCK_WIDTH 32
	int NumBlocks = N / BLOCK_WIDTH;
	//int NumBlocks2 = M / BLOCK_WIDTH;
	if ( N%BLOCK_WIDTH ) NumBlocks++;
	//if (M%BLOCK_WIDTH) NumBlocks2++;
		dim3 dimGrid(NumBlocks, NumBlocks);
	    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
		clock_t start = clock();
	matrix_mult<<< dimGrid, dimBlock >>>(dev_a, dev_b, dev_c, N);
	cudaMemcpy(&host_c, dev_c, sizeof(host_c), cudaMemcpyDeviceToHost);

	clock_t end = clock();
	float time = (float)(end - start) / CLOCKS_PER_SEC;
	printf("%f seconds\n", time);

	//for (int i = 0; i < N; i++)
	//for (int j = 0; j < N;j++)
		//printf("%f\n", host_c[i][j]);

	cudaFree(dev_a);//�ͷ�GPU�ڴ�
	cudaFree(dev_b);//�ͷ�GPU�ڴ�
	cudaFree(dev_c);//�ͷ�GPU�ڴ�

	

	return 0;
}

