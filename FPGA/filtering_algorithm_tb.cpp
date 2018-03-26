/**********************************************************************
* Felix Winterstein, Imperial College London
*
* File: filtering_algorithm_tb.cpp
*
* Revision 1.01
* Additional Comments: distributed under a BSD license, see LICENSE.txt
*
**********************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sds_lib.h>

#include "filtering_algorithm_top.h"
#include "filtering_algorithm_util.h"
#include "build_kdTree.h"


// recursively split the kd-tree into P sub-trees (P is parallelism degree)
void recursive_split(uint p,
                    uint n,
                    data_type bnd_lo,
                    data_type bnd_hi,
                    uint *idx,
                    data_type *data_points,
                    uint *i,
                    uint *ofs,
                    node_pointer *root,
                    kdTree_type *heap,
                    kdTree_type *tree_image,
                    node_pointer *tree_image_addr,
                    uint n0,
                    uint k,
                    double std_dev)
{
    if (p==P) {
        printf("Sub-tree %d: %d data points\n",*i,n);
        node_pointer rt = buildkdTree(data_points, idx, n, &bnd_lo, &bnd_hi, *i*HEAP_SIZE/2/P, heap);
        root[*i] = rt;
        uint offset = *ofs;
        //readout_tree(true, n0, k, std_dev, rt, heap, offset, tree_image, tree_image_addr);
        //do not write file
        readout_tree(false, n0, k, std_dev, rt, heap, offset, tree_image, tree_image_addr);
        *i = *i + 1;
        *ofs = *ofs + 2*n-1;
    } else {
        uint cdim;
        coord_type cval;
        uint n_lo;
        split_bounding_box(data_points, idx, n, &bnd_lo, &bnd_hi, &n_lo, &cdim, &cval);
        // update bounding box
        data_type new_bnd_hi = bnd_hi;
        data_type new_bnd_lo = bnd_lo;
        set_coord_type_vector_item(&new_bnd_hi.value,cval,cdim);
        set_coord_type_vector_item(&new_bnd_lo.value,cval,cdim);

        recursive_split(p*2, n_lo, bnd_lo, new_bnd_hi, idx, data_points,i,ofs,root, heap,tree_image,tree_image_addr,n0,k,std_dev);
        recursive_split(p*2, n-n_lo, new_bnd_lo, bnd_hi, idx+n_lo, data_points,i,ofs,root, heap,tree_image,tree_image_addr,n0,k,std_dev);
    }

}



int main()
{

    const uint n = 4096; // 16384
    const uint k = 5;   // 128
    const double std_dev = 0.75; //0.20

    //hardware variables
  /*  node_pointer root[P];
    coord_type_vector_ext wgtCent[HEAP_SIZE];
    coord_type_vector midPoint[HEAP_SIZE];
    coord_type_vector bnd_lo2[HEAP_SIZE];
    coord_type_vector bnd_hi2[HEAP_SIZE];
    coord_type_ext sum_sq[HEAP_SIZE];
    coord_type count[HEAP_SIZE];
    node_pointer left[HEAP_SIZE];
    node_pointer right[HEAP_SIZE];
    node_pointer tree_image_addr[HEAP_SIZE];
    coord_type_vector initial_centre_positions_simple[K];
    coord_type_vector clusters_out[K];
    coord_type_ext distortion_out[K];*/

    node_pointer *root;
    coord_type_vector_ext *wgtCent;
    coord_type_vector  *midPoint;
    coord_type_vector  *bnd_lo2;
    coord_type_vector  *bnd_hi2;
    coord_type_ext  *sum_sq;
    coord_type  *count;
    node_pointer  *left;
    node_pointer  *right;
    node_pointer  *tree_image_addr;
    coord_type_vector  *initial_centre_positions_simple;
    coord_type_vector  *clusters_out;
    coord_type_ext  *distortion_out;

    root = (node_pointer*) sds_alloc(P *sizeof(node_pointer));
    if(!root)
    {
    	printf("could not allocate root memory\n");
    	exit(0);
    }
    wgtCent = (coord_type_vector_ext*) sds_alloc(HEAP_SIZE *sizeof(coord_type_vector_ext));
    if(!wgtCent)
    {
    	printf("could not allocate wgtCent memory\n");
    	exit(0);
    }
    midPoint = (coord_type_vector*) sds_alloc(HEAP_SIZE *sizeof(coord_type_vector));
    if(!midPoint)
    {
    	printf("could not allocate midPoint memory\n");
    	exit(0);
    }
    bnd_lo2 = (coord_type_vector*) sds_alloc(HEAP_SIZE *sizeof(coord_type_vector));
    if(!bnd_lo2)
    {
    	printf("could not allocate bnd_lo2 memory\n");
    	exit(0);
    }
    bnd_hi2 = (coord_type_vector*) sds_alloc(HEAP_SIZE *sizeof(coord_type_vector));
    if(!bnd_hi2)
    {
    	printf("could not allocate bnd_hi2 memory\n");
    	exit(0);
    }
    sum_sq = (coord_type_ext*) sds_alloc(HEAP_SIZE *sizeof(coord_type_ext));
    if(!sum_sq)
    {
    	printf("could not allocate sum_sq memory\n");
    	exit(0);
    }
    count = (coord_type*) sds_alloc(HEAP_SIZE *sizeof(coord_type));
    if(!count)
    {
    	printf("could not allocate count memory\n");
    	exit(0);
    }
    left = (node_pointer*) sds_alloc(HEAP_SIZE *sizeof(node_pointer));
    if(!left)
    {
    	printf("could not allocate left memory\n");
    	exit(0);
    }
    right = (node_pointer*) sds_alloc(HEAP_SIZE *sizeof(node_pointer));
    if(!right)
    {
    	printf("could not allocate right memory\n");
    	exit(0);
    }
    tree_image_addr = (node_pointer*) sds_alloc(HEAP_SIZE *sizeof(node_pointer));
    if(!tree_image_addr)
    {
    	printf("could not allocate tree_image_addr memory\n");
    	exit(0);
    }
    initial_centre_positions_simple = (coord_type_vector*) sds_alloc(K *sizeof(coord_type_vector));
    if(!initial_centre_positions_simple)
    {
    	printf("could not allocate initial_centre_positions_simple memory\n");
    	exit(0);
    }
    clusters_out = (coord_type_vector*) sds_alloc(K *sizeof(coord_type_vector));
    if(!clusters_out)
    {
    	printf("could not allocate clusters_out memory\n");
    	exit(0);
    }
    distortion_out = (coord_type_ext*) sds_alloc(K *sizeof(coord_type_ext));
    if(!distortion_out)
    {
    	printf("could not allocate distortion_out memory\n");
    	exit(0);
    }

    /*uint *idx = new uint[N];
    data_type *data_points = new data_type[N];
    uint *cntr_indices = new uint[K];
    kdTree_type *heap = new kdTree_type[HEAP_SIZE];
    data_type *initial_centre_positions= new data_type[K];
    */
    uint idx[N];
    data_type data_points[N];
    uint cntr_indices[K];
    kdTree_type heap[HEAP_SIZE];
    data_type initial_centre_positions[K];


    // read data points from file
    if (read_data_points(n,k,std_dev,data_points,idx) == false)
        return 1;



    // read intial centre from file (random placement
//    if (read_initial_centres(n,k,std_dev,initial_centre_positions,cntr_indices) == false)
  //      return 1;

    // print initial centres
   /* printf("Initial centres\n");
    for (uint i=0; i<k; i++) {
        printf("%d: ",i);
        for (uint d=0; d<D-1; d++) {
            printf("%d ",get_coord_type_vector_item(initial_centre_positions[i].value, d).to_int());
        }
        printf("%d\n",get_coord_type_vector_item(initial_centre_positions[i].value, D-1).to_int());
    }*/

     printf("Initial centres\n");
        for (uint i=0; i<k; i++) {
        	initial_centre_positions_simple[i] = 0;
            printf("%d: ",i);
            for (uint d=0; d<D-1; d++) {

                printf("%d ",get_coord_type_vector_item(initial_centre_positions_simple[i], d).to_int());
            }
            printf("%d\n",get_coord_type_vector_item(initial_centre_positions_simple[i], D-1).to_int());
        }


    // compute axis-aligned hyper rectangle enclosing all data points
    data_type bnd_lo, bnd_hi;
    compute_bounding_box(data_points, idx, n, &bnd_lo, &bnd_hi);



    /*kdTree_type *tree_image = new kdTree_type[HEAP_SIZE];
    node_pointer *tree_image_addr = new node_pointer[HEAP_SIZE];*/

    kdTree_type tree_image[HEAP_SIZE];



    uint z=0;
    uint ofs=0;
    recursive_split(1, n, bnd_lo, bnd_hi, idx, data_points,&z,&ofs,root,heap,tree_image,tree_image_addr,n,k,std_dev);



    // FIXME: get automatic co-simulation working

    /*
    for (uint i=0; i<P; i++) {
    	root[i] = 0;
    }
    */
    for (uint i=0; i<2*n-1; i++) {
    	bnd_hi2[i] = tree_image[i].bnd_hi.value;
    	bnd_lo2[i] = tree_image[i].bnd_lo.value;
    	count[i] = tree_image[i].count.VAL;
    	left[i] = tree_image[i].left.VAL;
    	right[i]=tree_image[i].right.VAL;
    	midPoint[i]=tree_image[i].midPoint.value;
    	sum_sq[i]=tree_image[i].sum_sq.VAL;
    	wgtCent[i] = tree_image[i].wgtCent.value;
    }


    #define TIME_STAMP_INIT_HW  unsigned long long clock_start_hw, clock_end_hw;  clock_start_hw = sds_clock_counter();
    #define TIME_STAMP_HW  { clock_end_hw = sds_clock_counter(); printf("FPGA ON: Average number of processor cycles : %llu \n", (clock_end_hw-clock_start_hw)); clock_start_hw = sds_clock_counter();  }

    int max_iteration_count = 10;
    TIME_STAMP_INIT_HW
    filtering_algorithm_top(wgtCent,midPoint,bnd_lo2,bnd_hi2,sum_sq,count,left,right,tree_image_addr,initial_centre_positions_simple,2*n-1-1-(P-1),k,root,distortion_out,clusters_out,max_iteration_count);
    TIME_STAMP_HW

    // print initial centres
    printf("New centres after clustering\n");
    for (uint i=0; i<k; i++) {
        printf("%d: ",i);
        for (uint d=0; d<D-1; d++) {
            printf("%d ",get_coord_type_vector_item(clusters_out[i], d).to_int());
        }
        printf("%d\n",get_coord_type_vector_item(clusters_out[i], D-1).to_int());
    }


  /*  delete[] idx;
    delete[] data_points;
    delete[] initial_centre_positions;
    delete[] cntr_indices;

    delete[] heap;
    delete[] tree_image;
    delete[] tree_image_addr;*/


    return 0;
}
