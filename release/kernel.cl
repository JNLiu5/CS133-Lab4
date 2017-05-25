__kernel void conv(__global int *Cout,
			__global int *Cin,
			__global int *weight,
			__global int *bias) {                                                   
   // Get the work-item's unique ID            
   int idx = get_global_id(0);
	printf("%d\n", idx);
                                                    
   // Add the corresponding locations of            
   // 'A' and 'B', and store the result in 'C'.     
   // C[idx] = A[idx] + B[idx];
}                                                   
