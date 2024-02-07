import dask
from dask.distributed import Client as DaskClient
from dask.distributed import LocalCluster as DaskLocalCluster

class LocalDaskManager:
    _cluster = None
    _client  = None
    
    @classmethod
    def get_or_create_cluster(cls) -> dask.distributed.LocalCluster:
        if cls._cluster is None:
            cls._cluster = DaskLocalCluster()
        return cls._cluster
    
    @classmethod
    def get_or_create_client(cls) -> dask.distributed.Client:
        if cls._client is None:
            cls._client = DaskClient(cls.get_or_create_cluster())
        return cls._client