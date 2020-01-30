from ibapi.contract import Contract
from ibapi.order import Order


class IBContract(Contract):
    def __init__(self, **kwargs):
        """
        make Contract accept fields at init
        :param kwargs: fields of Contract
        """
        Contract.__init__(self)
        self.__dict__.update(**kwargs)


class IBOrder(Order):
    def __init__(self, **kwargs):
        """
        make Order accept fields at init
        :param kwargs: fields of Order
        """
        Order.__init__(self)
        self.__dict__.update(**kwargs)
