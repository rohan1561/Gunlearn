import numpy as np
import json
import os

#DIR = '../saves/saves_c10_random_95'
#DIR = '../saves/saves_c10_classwise_95'
#DIR = '../saves/saves_c10_random'
#DIR = '../saves/saves_c10_classwise'

#DIR = '../saves/saves_c100_classwise_95'
#DIR = '../saves/saves_c100_random'
#DIR = '../saves/saves_c100_classwise'

#DIR = './saves/saves_c10_random_95'
DIR = './saves/saves_c10_random'

{'accuracy': {'train': 87.11333333333333, 'retain': 96.73086419300974, 'forget': 0.5555555565092298, 'val': 0.1999999942779541, 'test': 91.32222222222222}, 'SVC_MIA_forget_efficacy': {'correctness': 0.9944444444444445, 'confidence': 1.0, 'entropy': 0.8088888888888889, 'm_entropy': 1.0, 'prob': 0.5933333333333333}, 'SVC_MIA_training_privacy': {'correctness': 0.5272361111111111, 'confidence': 0.5338611111111111, 'entropy': 0.5244444444444445, 'm_entropy': 0.5100833333333333, 'prob': 0.5113888888888889}}

trains = []
retains = []
forgets = []
vals = []
tests = []
corr = []
conf = []
ent = []
ment = []
prob = []
for root, dirs, files in os.walk(DIR, topdown=False):
    '''
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))
    '''
    files = list(filter(lambda x: x.endswith('.json'), files))
    for f in files:
        with open(os.path.join(root, f)) as f1:
            results = json.load(f1)
        trains.append(results['accuracy']['train'])
        retains.append(results['accuracy']['retain'])
        forgets.append(results['accuracy']['forget'])
        vals.append(results['accuracy']['val'])
        tests.append(results['accuracy']['test'])
        
        corr.append(results['SVC_MIA_forget_efficacy']['correctness'])
        conf.append(results['SVC_MIA_forget_efficacy']['confidence'])
        ent.append(results['SVC_MIA_forget_efficacy']['entropy'])
        ment.append(results['SVC_MIA_forget_efficacy']['m_entropy'])
        prob.append(results['SVC_MIA_forget_efficacy']['prob'])
 

print('mean')
print(np.mean(np.array(trains)), 'train')
print(np.mean(np.array(retains)), 'retain')
print(np.mean(np.array(forgets)), 'forget')
print(np.mean(np.array(vals)), 'val')
print(np.mean(np.array(tests)), 'test')
print('std')
print(np.std(np.array(trains)))
print(np.std(np.array(retains)))
print(np.std(np.array(forgets)))
print(np.std(np.array(vals)))
print(np.std(np.array(tests)))

print('xxxxxxxxxxxxxxxxxx')

print('mean')
print(np.mean(np.array(corr)), 'correctness')
print(np.mean(np.array(conf)), 'confidence')
print(np.mean(np.array(ent)), 'entropy')
print(np.mean(np.array(ment)), 'm_entropy')
print(np.mean(np.array(prob)), 'prob')
print('std')
print(np.std(np.array(corr)))
print(np.std(np.array(conf)))
print(np.std(np.array(ent)))
print(np.std(np.array(ment)))
print(np.std(np.array(prob)))

