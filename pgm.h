
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct _PGMData {
	int row;
	int col;
	int max_gray;
	char *matrix;
} PGMData;

char *allocate_dynamic_matrix(int row, int col)
{
	char *ret_val;
	

	ret_val = (char *)malloc(sizeof(char *) * row * col);
	if (ret_val == NULL) {
		perror("memory allocation failure");
		exit(EXIT_FAILURE);
	}

	return ret_val;
}

void deallocate_dynamic_matrix(char *matrix, int row)
{

	free(matrix);
}

#define HI(num) (((num) & 0x0000FF00) >> 8)
#define LO(num) ((num) & 0x000000FF)

void SkipComments(FILE *fp)
{
	int ch;
	char line[100];

	while ((ch = fgetc(fp)) != EOF && isspace(ch))
		;
	if (ch == '#') {
		fgets(line, sizeof(line), fp);
		SkipComments(fp);
	}
	else
		fseek(fp, -1, SEEK_CUR);
}

PGMData* readPGM(const char *file_name, PGMData *data)
{
	FILE *pgmFile;
	char version[3];
	int i;
	int lo, hi;

	pgmFile = fopen(file_name, "rb");
	if (pgmFile == NULL) {
		perror("cannot open file to read");
		exit(EXIT_FAILURE);
	}

	fgets(version, sizeof(version), pgmFile);
	if (strcmp(version, "P5")) {
		fprintf(stderr, "Wrong file type!\n");
		exit(EXIT_FAILURE);
	}

	SkipComments(pgmFile);
	fscanf(pgmFile, "%d", &data->col);
	SkipComments(pgmFile);
	fscanf(pgmFile, "%d", &data->row);
	SkipComments(pgmFile);
	fscanf(pgmFile, "%d", &data->max_gray);
	fgetc(pgmFile);

	data->matrix = allocate_dynamic_matrix(data->row, data->col);
	if (data->max_gray > 255)
		for (i = 0; i < data->row * data->col; ++i){

			hi = fgetc(pgmFile);
			lo = fgetc(pgmFile);
			data->matrix[i] = (hi << 8) + lo;
		}
	else
		for (i = 0; i < data->row * data->col; ++i){

			lo = fgetc(pgmFile);
			data->matrix[i] = lo;
		}

	fclose(pgmFile);
	return data;

}