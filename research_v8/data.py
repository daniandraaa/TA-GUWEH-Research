"""Matched real crops and one replacement per full K4 artificial bag."""
from collections import Counter
import numpy as np
import torch
from research_v4.data import PatientBags
from research_v6.data import load_development


class AugmentedBags(PatientBags):
    def __init__(self,X,table,ids,scaler,arm,fit_ids,synthetic,synthetic_labels,seed,size=4):
        super().__init__(X,table,ids,scaler,size,sample_seed=seed+10000)
        if arm not in {'real_only','real_aug','gan_aug'} or size!=4:raise ValueError('Fixed K4 arms required')
        self.arm=arm
        self.slot_rng=torch.Generator().manual_seed(seed+11000)
        self.source_rng=torch.Generator().manual_seed(seed+12000)
        self.synthetic=synthetic
        self.fake_groups={c:np.flatnonzero(np.asarray(synthetic_labels)==c) for c in range(3)}
        self.real_groups={c:[] for c in range(3)}
        if not set(fit_ids)<=set(ids):raise ValueError('Augmentation source must be training GAN-fit only')
        for hid in sorted(fit_ids):
            indices=np.flatnonzero(table.hospital_id.eq(hid).to_numpy())
            if not len(indices):raise ValueError('Missing real donor')
            self.real_groups[int(table.iloc[indices[0]].label)].append((hid,indices))
        if any(len(g)<2 for g in self.real_groups.values()) or any(not len(g) for g in self.fake_groups.values()):
            raise ValueError('All classes need real donor alternatives and synthetic windows')

    def rng_states(self):
        return {'crop':self.generator.get_state(),'slot':self.slot_rng.get_state(),'source':self.source_rng.get_state()}

    def restore_rng_states(self,state):
        self.generator.set_state(state['crop']);self.slot_rng.set_state(state['slot']);self.source_rng.set_state(state['source'])

    def __getitem__(self,item):
        # Parent generator only controls native crop choice; augmentations cannot shift it.
        value,label,hid=super().__getitem__(item)
        n=len(value);eligible=n==4
        meta={'hospital_id':hid,'label':label,'bag_windows':n,'eligible':eligible,
            'replacements':0,'slot':None,'donor_hospital_id':None,'donor_segment_index':None,'synthetic_index':None}
        if eligible:
            slot=int(torch.randint(4,(),generator=self.slot_rng))
            if self.arm!='real_only':
                if self.arm=='real_aug':
                    group=[g for g in self.real_groups[label] if g[0]!=hid]
                    donor,indices=group[int(torch.randint(len(group),(),generator=self.source_rng))]
                    index=int(indices[int(torch.randint(len(indices),(),generator=self.source_rng))])
                    replacement=np.array(self.X[index],dtype=np.float32,copy=True)
                    meta.update(donor_hospital_id=donor,donor_segment_index=index)
                else:
                    group=self.fake_groups[label]
                    index=int(group[int(torch.randint(len(group),(),generator=self.source_rng))])
                    replacement=np.array(self.synthetic[index],dtype=np.float32,copy=True)
                    meta['synthetic_index']=index
                replacement=(replacement-self.mean)/self.scale
                if not np.isfinite(replacement).all():raise ValueError('Nonfinite augmentation')
                value[slot]=torch.from_numpy(replacement)
                meta.update(replacements=1,slot=slot)
        return value,label,hid,meta


def collate(items):
    maximum=max(len(x) for x,_,_,_ in items)
    values=torch.zeros(len(items),maximum,*items[0][0].shape[1:]);mask=torch.zeros(len(items),maximum,dtype=torch.bool)
    for i,(x,_,_,_) in enumerate(items):values[i,:len(x)],mask[i,:len(x)]=x,True
    return values,torch.tensor([y for _,y,_,_ in items]),mask,[hid for _,_,hid,_ in items],[meta for _,_,_,meta in items]
