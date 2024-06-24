import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution

    def inverse_CDF(u: float):

        if (distribution == "cauchy"):
            return (kwargs['gamma'] * np.tan(np.pi * (u - 0.5)) + kwargs['peak_x'])
        elif (distribution == "exponential"):
            return (np.log(1 - u) / (-kwargs['lambda']))

    # 1. Generate a random number u from the standard uniform distribution in the interval [0,1], i.e. from 𝑈∼Unif[0,1]. 
    # 2. pass it through the generalized inverse of the desired CDF, i.e. 𝐹𝑋−1(𝑢)
    for i in range(num_samples):
        u = np.random.rand()
        samples.append(inverse_CDF(u))
    # 3. Round
    samples = list(np.round(samples, 4))
    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        plt.hist(samples)
        plt.xlabel(distribution + " distribution")
        plt.savefig(distribution + ".png")
        plt.show()
        # END TODO
