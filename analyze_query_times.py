times_ms = [7813, 8320, 8287, 8598, 8409, 496, 6276, 8305, 8318, 8783, 8601, 587, 8641, 8767, 9298, 146, 9146, 9297, 9556, 9284, 9244, 9224, 9303, 9235, 9951, 223, 37, 10894, 9328, 9224, 9211, 9319, 9473, 447, 119, 11227, 9481, 36, 46, 9234, 49, 179, 8021, 942, 85, 10862, 10788, 9144, 9284, 10141, 10773, 9242, 8625, 9493]

# 转换为秒
times_s = [t/1000.0 for t in times_ms]

total_queries = len(times_s)
total_time_s = sum(times_s)
avg_time_s = total_time_s / total_queries
min_time_s = min(times_s)
max_time_s = max(times_s)

print(f'查询总数: {total_queries}')
print(f'总搜索时间: {total_time_s:.1f} 秒')
print(f'平均每个查询时间: {avg_time_s:.2f} 秒')
print(f'最快查询时间: {min_time_s:.3f} 秒')
print(f'最慢查询时间: {max_time_s:.2f} 秒')
print(f'中位数时间: {sorted(times_s)[total_queries//2]:.2f} 秒')

# 分类统计
very_fast = [t for t in times_s if t < 1]  # 小于1秒
fast = [t for t in times_s if 1 <= t < 5]  # 1-5秒
normal = [t for t in times_s if 5 <= t < 10]  # 5-10秒
slow = [t for t in times_s if t >= 10]  # 大于等于10秒

print(f'\n时间分布:')
print(f'非常快 (<1秒): {len(very_fast)} 个查询, 平均 {sum(very_fast)/len(very_fast):.3f} 秒' if very_fast else '非常快 (<1秒): 0 个查询')
print(f'较快 (1-5秒): {len(fast)} 个查询, 平均 {sum(fast)/len(fast):.2f} 秒' if fast else '较快 (1-5秒): 0 个查询')
print(f'正常 (5-10秒): {len(normal)} 个查询, 平均 {sum(normal)/len(normal):.2f} 秒' if normal else '正常 (5-10秒): 0 个查询')
print(f'较慢 (>=10秒): {len(slow)} 个查询, 平均 {sum(slow)/len(slow):.2f} 秒' if slow else '较慢 (>=10秒): 0 个查询') 