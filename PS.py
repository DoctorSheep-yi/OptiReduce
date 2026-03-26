# PS.py

def shard_data(data, num_workers):
    shard_size = len(data) // num_workers
    shards = []

    for i in range(num_workers):
        start = i * shard_size
        if i == num_workers - 1:
            end = len(data)
        else:
            end = (i + 1) * shard_size

        shards.append(data[start:end])

    return shards


def combine_results(worker_results, method="average"):
    if not worker_results:
        return None

    if method == "sum":
        return sum(worker_results)

    elif method == "average":
        return sum(worker_results) / len(worker_results)

    else:
        raise ValueError("Unsupported method")
