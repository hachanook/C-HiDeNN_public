import jax.numpy as jnp


def get_material_model(config, device_main):
    if config["MATERIAL"]["Type"] == "Elastic":
        E = float(config["MATERIAL"]["E"])
        nu = float(config["MATERIAL"]["nu"])

        lmd = E*nu/((1+nu)*(1-2*nu)) # lame constant, lambda
        mu = E/2/(1+nu) # lame constant, mu, shear modulus
        D = E/((1+nu)*(1-2*nu)) * jnp.array([[1-nu, nu, nu, 0,0,0],
                                        [nu, 1-nu, nu, 0,0,0],
                                        [nu, nu, 1-nu, 0,0,0],
                                        [0,0,0,(1-2*nu)/2,0,0],
                                        [0,0,0,0,(1-2*nu)/2,0],
                                        [0,0,0,0,0,(1-2*nu)/2]], dtype=jnp.float64, device=device_main)
    return D
