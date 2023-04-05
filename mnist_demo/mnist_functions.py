

def create_benchmark_seq_batches(dataset_path, batch_size, num_batches, batch_compositions, position_only=True, seed=None):
    set_seeds() if seed == None else set_seeds(seed)
    if dataset_path.endswith('speed/'):
        dataloaders, dataset_sizes = create_dataloaders(dataset_path, batch_size)
    elif dataset_path.endswith('exoromper/'):
        dataloaders, dataset_sizes = create_dataloaders(dataset_path, batch_size, dataset_name='exoromper')
    
    test_seq = torch.empty((num_batches, batch_size, 3, 224, 224))
    if position_only:
        labels   = torch.empty((num_batches, batch_size, 3))
    else:
        labels   = torch.empty((num_batches, batch_size, 7))
    fnames = [["" for i in range(batch_size)] for j in range(num_batches)]
    
    for b in range(num_batches):
        idx_in_batch = 0
        for dn in batch_compositions[b]:
            dset = dataloaders[dn].dataset
            for i in range(batch_compositions[b][dn]):
                img, lbl, fnm = dset.__getitem__(randint(0, len(dset)-1))
                test_seq[b,idx_in_batch,:,:,:] = img
                labels[b,idx_in_batch,:] = torch.from_numpy(lbl)
                fnames[b][idx_in_batch] = fnm
                idx_in_batch += 1
        if not(idx_in_batch == batch_size):
            raise ValueError("Sum of batch compositions %i does not equal batch_size %i",(idx_in_batch, batch_size))

    return test_seq, labels, fnames