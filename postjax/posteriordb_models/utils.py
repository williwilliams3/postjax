
from posteriordb import PosteriorDatabase


def get_posterior(name, pdb_path="posteriordb/posterior_database"):
    try:
        my_pdb = PosteriorDatabase(pdb_path)
        posterior = my_pdb.posterior(name)
        return posterior
    except:
        raise Exception("Unable to load posteriordb model")