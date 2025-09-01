# dstree

# TODO list
- [x] clear shared_ptr abuse
- [ ] resolve the conflicts between multithreading and SIMD
- [ ] enable compiling on macOS, i.e., auto-switch to LibTorch-cpu


训练阶段：query=1k
nohup ./dstree --db_filepath /mnthdd/data/indexing/deep1b/deep1b-96-25m.bin --query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-1k.bin --query_size 1000 --series_length 96 --db_size 25000000 --leaf_size 10000 --exact_search --n_nearest_neighbor 1 --require_neurofilter --filter_train_is_gpu --filter_infer_is_gpu --filter_query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-10k.bin --filter_train_nexample 500 --learning_rate 0.01 --filter_train_min_lr 0.000001 --filter_train_clip_grad --filter_train_nepoch 300 --filter_train_mthread --filter_collect_nthread 16 --filter_train_nthread 16 --filter_remove_square --filter_is_conformal --filter_train_val_split 0.6 --filter_model_setting mlp --filter_conformal_recall 0.9 --filter_allocate_is_gain --device_id 0 --dump_index --index_dump_folderpath /home/qwang/projects/dstree/index_dump_25M_q1k --log_filepath /home/qwang/projects/dstree/log/train_check_prune_ratio_dstree_Leaf10k_q10_db25m.log > train_check_prune_ratio_dstree_Leaf10k_q10_db25m.out 2>&1 &


训练阶段：query=10
nohup ./dstree --db_filepath /mnthdd/data/indexing/deep1b/deep1b-96-25m.bin --query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-10.bin --query_size 10 --series_length 96 --db_size 25000000 --leaf_size 10000 --exact_search --n_nearest_neighbor 1 --require_neurofilter --filter_train_is_gpu --filter_infer_is_gpu --filter_query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-10k.bin --filter_train_nexample 500 --learning_rate 0.01 --filter_train_min_lr 0.000001 --filter_train_clip_grad --filter_train_nepoch 300 --filter_train_mthread --filter_collect_nthread 16 --filter_train_nthread 16 --filter_remove_square --filter_is_conformal --filter_train_val_split 0.6 --filter_model_setting mlp --filter_conformal_recall 0.9 --filter_allocate_is_gain --device_id 0 --dump_index --index_dump_folderpath /home/qwang/projects/dstree/index_dump_25M_q10 --log_filepath /home/qwang/projects/dstree/log/train_check_prune_ratio_dstree_Leaf10k_q10_db25m.log > train_check_prune_ratio_dstree_Leaf10k_q10_db25m.out 2>&1 &

测试查询阶段：
./dstree --db_filepath /mnthdd/data/indexing/deep1b/deep1b-96-25m.bin --query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-1k.bin --series_length 96 --db_size 25000000 --query_size 1000 --leaf_size 100000 --exact_search --n_nearest_neighbor 1 --require_neurofilter --filter_train_is_gpu --filter_infer_is_gpu --filter_query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-10k.bin --filter_train_nexample 500 --learning_rate 0.01 --filter_train_min_lr 0.000001 --filter_train_clip_grad --filter_train_nepoch 300 --filter_train_mthread --filter_collect_nthread 16 --filter_train_nthread 16 --filter_remove_square --filter_is_conformal --filter_train_val_split 0.6 --filter_model_setting mlp --filter_conformal_recall 0.9 --filter_allocate_is_gain --device_id 0 --load_index --load_filters --index_load_folderpath /home/qwang/projects/dstree/index_dump_25M_q10 --log_filepath /home/qwang/projects/dstree/log/test_lb_prune_ratio_dstree_Leaf10k_q1k_db25m_0901.log 



测试查询deep1b-96-10m-test-0.2-1k.bin的结果：

1）同时打开filter+lower bound：
Overall Lower Bound Pruning Statistics:
Total series that could be visited: 25000000000
Total series pruned by lower bound: 1557700727
Average lower bound pruning rate: 0.0623 (6.23%)
[*** LOG ERROR #0001 ***] [2025-09-01 11:10:20] [file_logger_mt] {invalid type specifier}
[MAIN] Search completed (49048 ms)

2）如果只测试lower bound的效果：
Overall Lower Bound Pruning Statistics:
Total series that could be visited: 25000000000
Total series pruned by lower bound: 3098324882
Average lower bound pruning rate: 0.1239 (12.39%)
[*** LOG ERROR #0001 ***] [2025-09-01 13:48:16] [file_logger_mt] {invalid type specifier}
[MAIN] Search completed (5930973 ms)
