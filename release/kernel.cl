#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112
#define KERNEL 5

__kernel
void conv(__global float* Cout, 
          __global float* Cin, 
          __global float* weight, 
          __global float* bias) {
    int i = get_global_id(0);
    float C[IMROW][IMROW];

    for(int h = 0; h < IMROW; h++) {
        for(int w = 0; w < IMROW; w++) {
            C[h][w] = bias[i];
        }
    }

    // Convolution
    for(int j = 0; j < NUM; j++) {
        for(int h = 0; h < IMROW; h++) {
            for(int w = 0; w < IMROW; w++) {
                for(int p = 0; p < KERNEL; p++) {
                    for(int q = 0; q < KERNEL; q++)
                        C[h][w] += weight[i*NUM*KERNEL*KERNEL + j*KERNEL*KERNEL + p*KERNEL + q] * Cin[j*INIMROW*INIMROW + (h + p)*INIMROW + (w + q)];
                }
            }
        }
    }

    // ReLU
    for (int h = 0; h < IMROW; h++) {
        for (int w = 0; w < IMROW; w++) {
            C[h][w] = fmax((float)0, C[h][w]);
        }
    }

    // Max pooling
    for (int h = 0; h < OUTIMROW; h++) {
        for (int w = 0; w < OUTIMROW; w++) {
            float local_max = C[2 * h][2 * w];
            local_max = fmax(local_max, C[2 * h + 1][2 * w]);
            local_max = fmax(local_max, C[2 * h + 1][2 * w + 1]);
            local_max = fmax(local_max, C[2 * h][2 * w + 1]);
            Cout[i*OUTIMROW*OUTIMROW + h*OUTIMROW + w] = local_max;
        }
    }
}
