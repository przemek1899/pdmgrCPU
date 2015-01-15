#include <stdio.h>
#include "readFiles.h"

void readBinaryMatrix(double * data, char * filename, const unsigned int totalByteSize, size_t size){

    FILE * pf = fopen(filename, "rb");
    if(pf != NULL){
        unsigned int readBytes = 0;
        while(totalByteSize - readBytes){
            readBytes += fread((void*)(data+readBytes), size, totalByteSize - readBytes, pf);
        }
        fclose(pf);
    }
}
