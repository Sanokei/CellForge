"""Native niche adapters for independently generated structural task families."""
def make_niche(base,module):
    class StructuralNiche(base.Niche):
        name,template,general,verify=module.NICHENAME,module.TEMPLATE,True,True
        def patterns(self,limit,seed):
            return [{'id':self.name+'-%d'%i,'niche':self.name,'truth':module.ORACLE_PARAMS,
                     'apply':module.truth,'sample_input':lambda rng,i=i:module.generated({'index':i},rng.randint(1,1000000))['input']}
                    for i in range(limit)]
    return StructuralNiche()
