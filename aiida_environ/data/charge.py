# -*- coding: utf-8 -*-
from typing import Tuple

from aiida.orm import Data


class EnvironCharge:
    def __init__(self, **kwargs):
        self._charge = None
        self._position = None
        self._spread = None
        self._dim = None
        self._axis = None

        if "environ_charge" in kwargs:
            environ_charge = kwargs.pop("environ_charge")
            if kwargs:
                raise ValueError(
                    "If you pass 'environ_charge', you cannot pass any further parameter to the EnvironCharge constructor"
                )
            if not isinstance(environ_charge, EnvironCharge):
                raise ValueError("'environ_charge' must be of type EnvironCharge")
            self.charge = environ_charge.charge
            self.position = environ_charge.position
            self.spread = environ_charge.spread
            self.dim = environ_charge.dim
            self.axis = environ_charge.axis
        elif "raw" in kwargs:
            raw = kwargs.pop("raw")
            if kwargs:
                raise ValueError(
                    "If you pass 'raw', you cannot pass any further parameter to the Site constructor"
                )
            try:
                self.charge = raw["charge"]
                self.position = raw["position"]
                self.spread = raw["spread"]
                self.dim = raw["dim"]
                self.axis = raw["axis"]
            except KeyError as exc:
                raise ValueError(
                    f"Invalid raw object, it does not contain any key {exc.args[0]}"
                )
            except TypeError:
                raise ValueError("Invalid raw object, it is not a dictionary")
        else:
            try:
                self.charge = kwargs.pop("charge")
                self.position = kwargs.pop("position")
                self.spread = kwargs.pop("spread")
                self.dim = kwargs.pop("dim")
                self.axis = kwargs.pop("axis")
            except KeyError as exc:
                raise ValueError(f"You need to specify {exc.args[0]}")
            if kwargs:
                raise ValueError(f"Unrecognized parameters: {kwargs.keys}")

    def get_raw(self):
        """
        Return the raw version of the site, mapped to a suitable dictionary.
        This is the format that is actually used to store each site of the
        structure in the DB.

        :return: a python dictionary with the site.
        """
        return {
            "charge": self.charge,
            "position": self.position,
            "spread": self.spread,
            "dim": self.dim,
            "axis": self.axis,
        }

    @property
    def charge(self):
        """Return the charge

        Returns:
            float: charge in electrons
        """
        return self._charge

    @charge.setter
    def charge(self, value: float):
        """Set the charge

        Args:
            value (float): charge in electrons
        """
        self._charge = float(value)

    @property
    def position(self):
        """Return the position

        Returns:
            Tuple[float]: coordinates in angstrom
        """
        return self._position

    @position.setter
    def position(self, value: Tuple[float]):
        """Set the position

        Args:
            value (Tuple[float]): coordinates in angstrom

        Raises:
            ValueError: length of tuple incorrect
            ValueError: list/tuple not provided
        """
        try:
            internal_pos = tuple(float(i) for i in value)
            if len(internal_pos) != 3:
                raise ValueError
        # value is not iterable or elements are not floats or len != 3
        except (ValueError, TypeError):
            raise ValueError(
                "Wrong format for position, must be a list of three float numbers."
            )
        self._position = internal_pos

    @property
    def spread(self):
        """Return the spread of the charge

        Returns:
            float: spread
        """
        return self._spread

    @spread.setter
    def spread(self, value: float):
        """Set the spread of the charge

        Args:
            value (float): spread
        """
        self._spread = float(value)

    @property
    def dim(self):
        """Return the dimensionality of the charge object

        Returns:
            int: dimensionality (0-2)
        """
        return self._dim

    @dim.setter
    def dim(self, value: int):
        """Set the charge

        Args:
            value (int): dimensionality (0-2)
        """
        if value < 0 or value > 2:
            raise ValueError("Dimensionality must be between 0 and 2")
        self._dim = int(value)

    @property
    def axis(self):
        """Return the axis (1-3), where x=1, y=2, z=3

        If dim=2, the axis is orthogonal to the 2D plane
        If dim=1, the axis is along the 1D direction

        Returns:
            int: axis
        """
        return self._axis

    @axis.setter
    def axis(self, value: int):
        """Return the axis (1-3), where x=1, y=2, z=3

        If dim=2, the axis is orthogonal to the 2D plane
        If dim=1, the axis is along the 1D direction

        Args:
            value (int): axis
        """
        if value < 1 or value > 3:
            raise ValueError("Axis must be between 1 and 3")
        self._axis = int(value)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {str(self)}>"

    def __str__(self):
        ax = {1: "x", 2: "y", 3: "z"}
        return (
            f"charge '{self.charge}' @ {self.position[0]},{self.position[1]},{self.position[2]}"
            f" (dim {self.dim}, '{ax[self.axis]}' axis, spread={self.spread})"
        )


class EnvironChargeData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def append_charge(
        self, charge: float, position: Tuple[float], spread: float, dim: int, axis: int
    ):
        charge = EnvironCharge(
            charge=charge, position=position, spread=spread, dim=dim, axis=axis
        )
        self.attributes.setdefault("environ_charges", []).append(charge.get_raw())

    def clear_charges(self):
        """
        Removes all charges for the EnvironChargeData object.
        """
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed(
                "The EnvironChargeData object cannot be modified, it has already been stored"
            )

        self.set_attribute("environ_charges", [])

    @property
    def environ_charges(self):
        """
        Returns a list of sites.
        """
        try:
            raw_charges = self.get_attribute("environ_charges")
        except AttributeError:
            raw_charges = []
        return [EnvironCharge(raw=i) for i in raw_charges]

    def environ_output(self):
        """Prints out to string for `environ.in`"""

        environ_charges = self.environ_charges
        if len(environ_charges) == 0:
            # nothing here, just return empty string
            return ""

        # TODO add support for other units
        inputappend = "EXTERNAL_CHARGES (angstrom)\n"
        for charge in environ_charges:
            inputappend += (
                f"{charge.charge} {charge.position[0]:10.6f} {charge.position[1]:10.6f} {charge.position[2]:10.6f} "
                f"{charge.spread:10.6f} {charge.dim:d} {charge.axis:d}\n"
            )

        return inputappend

    def __len__(self):
        return len(self.environ_charges)
