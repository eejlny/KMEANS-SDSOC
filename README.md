# KMEANS-SDSOC

Test of KMEANS SDSOC using a ZedBoard. 

This algorithm performance is data dependent if early termination is defined. 

These tests are based in SDSOC 2016.3. 
Tests using 2017.4 fail and output is stuck at zero (reason unknown). 

Implementation parameters are controlled in filtering_algorithm_top.h:

#define SYNDATA //define this to use some simple data for testinginstead of premade files.

#define D 3         // data dimensionality (sdsoc can only do D 3 at the moment) notice typedef ap_int<(D+1)*COORD_BITWIDTH> coord_type_vector the D+1 makes the width 64 bit so SDSOC can handle it.
So if D 2,4 ... then ap_int<(D)..
So if D 3,5 ... then ap_int<(D+1)..
Otherwise software and hardware port width differ and implementation error.

#define N 4096//128 //32768     // max number of data points

#define K 10        // max number of centres

#define L 30         // max number of outer clustering iterations

#define EARLY_TERMINATION  //exit when algorithm stops converging so that L is not really reached if convergence takes place.

The run parameters are adjusted in the main since the implementation parameters defined max values. 

With these implementation parameters memory utilization is about 85% in the zedboard. Doubling the number of data points will exceed the available memory. 

Performance (in CPU clock cycles with k 2 and n 4096 in main):

ET (Early termination)

CPU version ET : 11,410,562

FPGA version ET : 1,843,768

Two centres are found at 50,50,50,and 150,150,150 using the SYNDATA that splits the data space in 3. 


