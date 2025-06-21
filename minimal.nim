import nimpy

type MyGaussian = ref object of PyNimObjectExperimental
    mu: float
    sigma: float

proc initMyGaussian(self: MyGaussian, mu: float, sigma: float) {.exportpy: "__init__".} =
    self.mu = mu
    self.sigma = sigma

proc `$`(self: MyGaussian): string {.exportpy: "__repr__".} =
    "MyGaussian"

proc `+`(self: MyGaussian, other: MyGaussian): MyGaussian {.exportpy: "__add__".} =
    MyGaussian(mu: self.mu + other.mu, sigma: self.sigma + other.sigma)

when isMainModule:
    discard