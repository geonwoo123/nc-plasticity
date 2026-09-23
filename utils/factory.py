def get_model(model_name, args):
    name = model_name.lower()

    if name == "sec":
        from models.sec import Learner
    elif name == "nc_plasticity":
        from models.nc_plasticity import Learner
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Learner(args)
