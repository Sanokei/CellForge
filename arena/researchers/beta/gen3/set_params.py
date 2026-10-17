import json, sys
alpha, beta, Gamma, s_boost = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
json.dump({"alpha": alpha, "beta": beta, "Gamma": Gamma, "s_boost": s_boost},
          open("/srv/cellforge-arena/researchers/beta/gen3/cal_params.json", "w"))
print("params set:", alpha, beta, Gamma, s_boost)
