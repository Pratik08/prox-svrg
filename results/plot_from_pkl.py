import pickle
import matplotlib.pyplot as plt

gd_nnz = 'sido_gd_nnz.pkl'
gd_loss = 'sido_gd_og.pkl'
gd_efp = 'sido_gd_efp.pkl'

prox_sag_nnz = 'sido_proxsag_nnz.pkl'
prox_sag_loss = 'sido_proxsag_og.pkl'
prox_sag_efp = 'sido_proxsag_efp.pkl'

prox_sg_nnz = 'sido_proxsg_nnz.pkl'
prox_sg_loss = 'sido_proxsg_og.pkl'
prox_sg_efp = 'sido_proxsg_efp.pkl'

prox_svrg_nnz = 'sido_proxsvrg_nnz.pkl'
prox_svrg_loss = 'sido_proxsvrg_og.pkl'
prox_svrg_efp = 'sido_proxsvrg_efp.pkl'

gd_nnz = pickle.load(open(gd_nnz, "rb"))
sag_nnz = pickle.load(open(prox_sag_nnz, "rb"))
sg_nnz = pickle.load(open(prox_sg_nnz, "rb"))
svrg_nnz = pickle.load(open(prox_svrg_nnz, "rb"))

plt.plot(gd_nnz)
plt.plot(sag_nnz)
plt.plot(sg_nnz)
plt.plot(svrg_nnz)
plt.savefig('ALL_NNZ.png')

gd_loss_list = pickle.load(open(gd_loss, "rb"))
sag_loss = pickle.load(open(prox_sag_loss, "rb"))
sg_loss = pickle.load(open(prox_sg_loss, "rb"))
svrg_loss = pickle.load(open(prox_svrg_loss, "rb"))

plt.plot(gd_loss_list)
plt.plot(sag_loss)
plt.plot(sg_loss)
plt.plot(svrg_loss)
plt.savefig('ALL_LOSS.png')
