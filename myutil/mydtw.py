from tslearn.metrics import dtw_path

def my_dtw_tdi(y_cpu, pred_cpu):
    total_sim,total_tdi = 0,0
    for i in range(0,y_cpu.shape[0]):
        path,sim = dtw_path(y_cpu[i,:], pred_cpu[i,:])
        total_sim += sim
        
        Dist = 0
        for i,j in path:
            Dist += (i-j)*(i-j)
        tdi = Dist / (y_cpu.shape[1]*y_cpu.shape[1])
        total_tdi += tdi    
    return total_sim/y_cpu.shape[0],total_tdi/y_cpu.shape[0]
    