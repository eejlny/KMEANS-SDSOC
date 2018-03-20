# KMEANS-SDSOC

Test of KMEANS SDSOC using a ZedBoard. This algorithm performance is data dependent if early termination is defined. 

These tests are based in SDSOC 2016.3. Tests using 2017.4 fail and output is stuck at zero (reason unknown). 

Implementation parameters are controlled in filtering_algorithm_top.h:

#define SYNDATA //define this to use some simple data for testinginstead of premade files.

#define D 3         // data dimensionality (sdsoc can only do D 3 at the moment) notice typedef ap_int<(D+1)*COORD_BITWIDTH> coord_type_vector the D+1 makes the width 64 bit so SDSOC can handle it.

#define N 4096//128 //32768     // max number of data points

#define K 2 //256       // max number of centres

#define L 30         // max number of outer clustering iterations

#define EARLY_TERMINATION  //exit when algorithm stops converging so that L is not really reached if convergence takes place.

With these parameters memory utilizatoin is about 50% in the zedboard. Performance is as follows (in CPU clock cycles):

NET (No early termination)

ET (Early termination)

CPU version NET :  87,849,048

CPU version ET : 11,267,774

FPGA version NET : 5,360,372

FPGA version ET : 1,288,384
