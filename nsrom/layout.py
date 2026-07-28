from dataclasses import dataclass
import os
@dataclass
class LocalROMLayout:
    base_dir: str            # e.g. "local_rom"
    n_clusters: int
    inner_product_type: str

    @property
    def root(self):        return f"{self.base_dir}/K{self.n_clusters}_{self.inner_product_type}"
    @property
    def clustering_dir(self):   return f"{self.root}/clustering"
    def cluster_dir(self, k):   return f"{self.root}/cluster_{k}"
    def pod_dir(self, k):       return f"{self.cluster_dir(k)}/pod_basis"
    def ops_dir(self, k):       return f"{self.cluster_dir(k)}/operators"
    def deim_basis(self, k):    return f"{self.cluster_dir(k)}/deim_basis.npz"
    def deim_ops(self, k):      return f"{self.cluster_dir(k)}/deim_ops.npz"
    def sweep_csv(self):        return f"{self.root}/sweep_results.csv"
    def pod_exists(self, k):    return os.path.isdir(self.pod_dir(k)) and os.path.isdir(self.ops_dir(k))
    def run_json(self):         return f"{self.root}/run.json"