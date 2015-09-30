#include <GL/glew.h>
#include "Dependencies\freeglut\freeglut.h"

#include <stdio.h>
#include <stdlib.h>

#include "pgm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdio.h>

// Interop with a pixel buffer 
// Runs both sobel and blur
// Testing keep map and fusion

int run = 1;

GLuint tex;
GLuint buf;

GLuint out_tex;
GLuint out_buf;

GLuint in_tex;
GLuint in_buf;

cudaArray *out_array;
cudaArray *in_array;

cudaArray *out_temp;
char *out_iqm;
cudaArray *in_temp;
char *in_iqm;

char* keepMap;


GLuint width;
GLuint height;


cudaGraphicsResource *out_resource;
cudaGraphicsResource *in_resource;

cudaGraphicsResource *resource;


// Reference the textures in CUDA kernels
texture<char, 2, cudaReadModeElementType> out_texRef;
texture<char, 2, cudaReadModeElementType> in_texRef;


// Used for checking errors
void checkError(int line) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s) at %d !\n", cudaGetErrorString(err), line);
		system("PAUSE");
		exit(EXIT_FAILURE);
	}
}


// Fuses the the two images using regular data accessing
__global__ void fuse2(char *keep, char *in, char *out, char* data) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (keep[x] == 1) {
		data[x] = in[x];
	}
	else {
		data[x] = out[x];
	}
}

// Fuses the two images using texture references
__global__ void fuse(char *keep, char* data, unsigned int pitch) {
	unsigned char *p =
		(unsigned char *)(((char *)data) + blockIdx.x*pitch);

	for (int i = threadIdx.x; i < pitch; i += blockDim.x) {
		if (keep[i] == 1) {
			p[i] = tex2D(in_texRef, (float)i, (float)blockIdx.x);
		}
		else {
			p[i] = tex2D(out_texRef, (float)i, (float)blockIdx.x);
		}
	}
}

// Generates the keep map using regular accessing
__global__ void keep2(char* keep, char *in, char *out, unsigned int pitch) {

	for (int i = threadIdx.x; i < pitch; i += blockDim.x) {
		if (in[i] > out[i]) {
			keep[i] = 1;
		}
		else {
			keep[i] = 0;
		}
	}
	
}

// Genereates the keep map using texture references
__global__ void keep(char *out, int w, int h, unsigned int pitch) {

	unsigned char *p =
		(unsigned char *)(((char *)out) + blockIdx.x*pitch);

	for (int i = threadIdx.x; i < w; i += blockDim.x) {
		if (tex2D(in_texRef, (float)i, (float)blockIdx.x) > tex2D(out_texRef, (float)i, (float)blockIdx.x)) {
			p[i] = 1;
		}
		else {
			p[i] = 0;
		}
	}

}

// Sobel operator kernel for the output texture reference
__global__ void out_sobel(char *original, int w, int h, unsigned int pitch) {


	unsigned char *p =
		(unsigned char *)(((char *)original) + blockIdx.x*pitch);

	for (int i = threadIdx.x; i < w; i += blockDim.x)
	{
		unsigned char ul = tex2D(out_texRef, (float)i - 1, (float)blockIdx.x - 1);
		unsigned char um = tex2D(out_texRef, (float)i + 0, (float)blockIdx.x - 1);
		unsigned char ur = tex2D(out_texRef, (float)i + 1, (float)blockIdx.x - 1);
		unsigned char ml = tex2D(out_texRef, (float)i - 1, (float)blockIdx.x + 0);
		unsigned char mm = tex2D(out_texRef, (float)i + 0, (float)blockIdx.x + 0);
		unsigned char mr = tex2D(out_texRef, (float)i + 1, (float)blockIdx.x + 0);
		unsigned char ll = tex2D(out_texRef, (float)i - 1, (float)blockIdx.x + 1);
		unsigned char lm = tex2D(out_texRef, (float)i + 0, (float)blockIdx.x + 1);
		unsigned char lr = tex2D(out_texRef, (float)i + 1, (float)blockIdx.x + 1);

		short Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
		short Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
		short Sum = (short)((abs((int)Horz) + abs((int)Vert)));

		if (Sum < 0)
		{
			p[i] = 0;
		}
		else if (Sum > 0xff)
		{
			p[i] = 0xff;
		}
		else {
			p[i] = (unsigned char)Sum;
		}
	}

}

// Sobel operator kernel for the input texture reference
__global__ void in_sobel(char *original, int w, int h, unsigned int pitch) {


	unsigned char *p =
		(unsigned char *)(((char *)original) + blockIdx.x*pitch);

	for (int i = threadIdx.x; i < w; i += blockDim.x)
	{
		unsigned char ul = tex2D(in_texRef, (float)i - 1, (float)blockIdx.x - 1);
		unsigned char um = tex2D(in_texRef, (float)i + 0, (float)blockIdx.x - 1);
		unsigned char ur = tex2D(in_texRef, (float)i + 1, (float)blockIdx.x - 1);
		unsigned char ml = tex2D(in_texRef, (float)i - 1, (float)blockIdx.x + 0);
		unsigned char mm = tex2D(in_texRef, (float)i + 0, (float)blockIdx.x + 0);
		unsigned char mr = tex2D(in_texRef, (float)i + 1, (float)blockIdx.x + 0);
		unsigned char ll = tex2D(in_texRef, (float)i - 1, (float)blockIdx.x + 1);
		unsigned char lm = tex2D(in_texRef, (float)i + 0, (float)blockIdx.x + 1);
		unsigned char lr = tex2D(in_texRef, (float)i + 1, (float)blockIdx.x + 1);

		short Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
		short Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
		short Sum = (short)((abs((int)Horz) + abs((int)Vert)));

		if (Sum < 0)
		{
			p[i] = 0;
		}
		else if (Sum > 0xff)
		{
			p[i] = 0xff;
		}
		else {
			p[i] = (unsigned char)Sum;
		}
	}

}

// NEEDS UPDATING, OVERFLOW BUG
// Blur kernel for the output texture reference
__global__ void out_blur(char *original, int w, int h, unsigned int pitch) {


	char *p = (((char *)original) + blockIdx.x*pitch);

	for (int i = threadIdx.x; i < w; i += blockDim.x)
	{
		float ul = ((float)tex2D(out_texRef, (float)i - 1, (float)blockIdx.x - 1)) * (float)(1.f / 9);
		float um = ((float)tex2D(out_texRef, (float)i + 0, (float)blockIdx.x - 1)) * (float)(1.f / 9);
		float ur = ((float)tex2D(out_texRef, (float)i + 1, (float)blockIdx.x - 1)) * (float)(1.f / 9);
		float ml = ((float)tex2D(out_texRef, (float)i - 1, (float)blockIdx.x + 0)) * (float)(1.f / 9);
		float mm = ((float)tex2D(out_texRef, (float)i + 0, (float)blockIdx.x + 0)) * (float)(1.f / 9);
		float mr = ((float)tex2D(out_texRef, (float)i + 1, (float)blockIdx.x + 0)) * (float)(1.f / 9);
		float ll = ((float)tex2D(out_texRef, (float)i - 1, (float)blockIdx.x + 1)) * (float)(1.f / 9);
		float lm = ((float)tex2D(out_texRef, (float)i + 0, (float)blockIdx.x + 1)) * (float)(1.f / 9);
		float lr = ((float)tex2D(out_texRef, (float)i + 1, (float)blockIdx.x + 1)) * (float)(1.f / 9);

		p[i] = (char)(ul + um + ur + ml + mm + mr + ll + lm + lr);
	}

}

// NEEDS UPDATING, OVERFLOW BUG
// Blur kernel for the input texture reference
__global__ void in_blur(char *original, int w, int h, unsigned int pitch) {


	char *p = (((char *)original) + blockIdx.x*pitch);

	for (int i = threadIdx.x; i < w; i += blockDim.x)
	{
		float ul = ((float)tex2D(in_texRef, (float)i - 1, (float)blockIdx.x - 1)) * (float)(1.f / 9);
		float um = ((float)tex2D(in_texRef, (float)i + 0, (float)blockIdx.x - 1)) * (float)(1.f / 9);
		float ur = ((float)tex2D(in_texRef, (float)i + 1, (float)blockIdx.x - 1)) * (float)(1.f / 9);
		float ml = ((float)tex2D(in_texRef, (float)i - 1, (float)blockIdx.x + 0)) * (float)(1.f / 9);
		float mm = ((float)tex2D(in_texRef, (float)i + 0, (float)blockIdx.x + 0)) * (float)(1.f / 9);
		float mr = ((float)tex2D(in_texRef, (float)i + 1, (float)blockIdx.x + 0)) * (float)(1.f / 9);
		float ll = ((float)tex2D(in_texRef, (float)i - 1, (float)blockIdx.x + 1)) * (float)(1.f / 9);
		float lm = ((float)tex2D(in_texRef, (float)i + 0, (float)blockIdx.x + 1)) * (float)(1.f / 9);
		float lr = ((float)tex2D(in_texRef, (float)i + 1, (float)blockIdx.x + 1)) * (float)(1.f / 9);

		p[i] = (char)(ul + um + ur + ml + mm + mr + ll + lm + lr);
	}

}



void initData() {
	_PGMData pic;
	//readPGM("lena512.pgm", &pic);
	readPGM("pgm/000.pgm", &pic);

	// Update width and height variables
	width = pic.col;
	height = pic.row;
	printf("%d %d \n", width, height);

	// Create channel description for cuda arrays
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<char>();

	// Setup cuda arrays for temp and output
	cudaMallocArray(&out_array, &channelDesc, pic.col, pic.row);
	cudaMemcpyToArray(out_array, 0, 0, pic.matrix, pic.col * pic.row * sizeof(char), cudaMemcpyHostToDevice);

	cudaMallocArray(&out_temp, &channelDesc, pic.col, pic.row);
	cudaMemcpyToArray(out_temp, 0, 0, pic.matrix, pic.col * pic.row * sizeof(char), cudaMemcpyHostToDevice);

	
	// Might need new reference to update?
	// _PGMData pic0;

	//readPGM("lena512.pgm", &pic);
	readPGM("pgm/001.pgm", &pic);

	// Setup cuda arrays for temp and input
	cudaMallocArray(&in_array, &channelDesc, pic.col, pic.row);
	cudaMemcpyToArray(in_array, 0, 0, pic.matrix, pic.col * pic.row * sizeof(char), cudaMemcpyHostToDevice);

	cudaMallocArray(&in_temp, &channelDesc, pic.col, pic.row);
	cudaMemcpyToArray(in_temp, 0, 0, pic.matrix, pic.col * pic.row * sizeof(char), cudaMemcpyHostToDevice);

	
	// Allocate space for other buffers
	cudaMalloc((void **)&keepMap, width * height * sizeof(char));
	cudaMalloc((void **)&out_iqm, width * height * sizeof(char));
	cudaMalloc((void **)&in_iqm, width * height * sizeof(char));
}

void initTexture() {

	// Initialize buffer pixel unpack buffer
	glGenBuffers(1, &buf);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf);
																										// GLBufferSubData for adding different data
	glBufferData(GL_PIXEL_UNPACK_BUFFER, height * width * sizeof(char), NULL, GL_STREAM_DRAW);			// This prevents unneccesary copy
	cudaGraphicsGLRegisterBuffer(&resource, buf, cudaGraphicsMapFlagsWriteDiscard);		// Registers resoure as device pointer

	// Initialize texture
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}





void display() {

	// Map data pointer to the final cuda resource for display
	char *data = NULL;
	size_t bytes;

	cudaGraphicsMapResources(1, &resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&data, &bytes, resource);
	
	// Bind cuda arrays to texture references
	cudaBindTextureToArray(out_texRef, out_array);
	cudaBindTextureToArray(in_texRef, in_array);

	// Run sobel kernels for output and input and copy data to buffers
	// Extra copy shouldn't be necessary when old images will be thrown away
	out_sobel << <height, 16 >> > (out_iqm, width, height, width);
	cudaMemcpyToArray(out_temp, 0, 0, out_iqm, height * width * sizeof(char), cudaMemcpyDeviceToDevice);
	in_sobel << <height, 16 >> > (in_iqm, width, height, width);
	cudaMemcpyToArray(in_temp, 0, 0, in_iqm, height * width * sizeof(char), cudaMemcpyDeviceToDevice);

	// Unbind texture references and bind to new buffers
	// Again shouldn't be necessary in the future
	cudaUnbindTexture(out_texRef);
	cudaUnbindTexture(in_texRef);
	cudaBindTextureToArray(out_texRef, out_temp);
	cudaBindTextureToArray(in_texRef, in_temp);

	// Run blur kernels and write to first buffer
	out_blur << <height, 16 >> > (out_iqm, width, height, width);
	in_blur << <height, 16 >> > (in_iqm, width, height, width);

	// Run keep map kernel to generate keep map
	keep2 <<< height, 16 >>> (keepMap, in_iqm, out_iqm, width);

	// Unbind texture references and bind to first buffers
	// Again shouldn't be necessary in the future
	cudaUnbindTexture(out_texRef);
	cudaUnbindTexture(in_texRef);
	cudaBindTextureToArray(out_texRef, out_array);
	cudaBindTextureToArray(in_texRef, in_array);

	// Run fuse kernel and write to the mapped data buffer
	fuse << < height, 16 >> > (keepMap, data, width);

	// Unbind texture references
	cudaUnbindTexture(out_texRef);
	cudaUnbindTexture(in_texRef);
	
	// Unmap resources
	cudaGraphicsUnmapResources(1, &resource, 0); // THIS NEEDS TO BE DONE TO USE IT WITH OPENGL

	// Go through OpenGL drawing to quads
	// Can update this at some point to used modern OpenGL and shaders
	glClear(GL_COLOR_BUFFER_BIT);

	glBindTexture(GL_TEXTURE_2D, tex);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 1.0); glVertex3f(1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 0.0); glVertex3f(1.0, 1.0, 0.5);
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, 1.0, 0.5);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);

	glDisable(GL_TEXTURE_2D);

	glutSwapBuffers();
	glutPostRedisplay(); // Creates continuous loop
}

int main(int argc, char **argv) {

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	// Add required version for openGL if necessary
	glutInitWindowPosition(600, 100);
	glutInitWindowSize(1024, 800);	// Should set this to be width and height, possibly with constants
	glutCreateWindow("OpenGL First Window");

	glewInit();

	// Initializations
	initData();
	initTexture();

	// OpenGL setup
	glutDisplayFunc(display);
	glutMainLoop();

	//getchar();
	return 0;
}