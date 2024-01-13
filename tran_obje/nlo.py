from tran_obje.sci import SciTran, SciObje


class NloTran(SciTran):
    def __init__(self, date, num_com, tran_type):
        super().__init__(date, num_com, tran_type)


class NloObje(SciObje):
    def __init__(self, date, num_com):
        super().__init__(date, num_com)

    def get_obje(self, weights, *args, **kwargs):
        obje = super().get_obje(weights)

        return obje

