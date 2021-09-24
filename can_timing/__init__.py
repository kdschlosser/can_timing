
try:
    import micropython
except ImportError:
    class micropython(object):

        @staticmethod
        def native(func):
            return func


class _TimingConst(object):
    """
    Constants used to calculate the bit timing registers

    If wanting to expand the bitrate calculations so it will support a
    different interface family this class would need to be subclassed
    and the approptiate values set to properly calculate the timing registers
    for that interface family.
    """

    tseg1_min = 4
    tseg1_max = 16
    tseg2_min = 2
    tseg2_max = 8
    sjw_max = 4
    brp_min = 1
    brp_max = 64
    brp_inc = 1
    fsys = 16000000
    additional_timings = ()


class _MCP251xConst(_TimingConst):
    tseg1_min = 3


class _ESP32Const(_TimingConst):
    tseg1_min = 2
    tseg2_min = 1
    brp_min = 2
    brp_max = 128
    brp_inc = 2
    fsys = 80000000


class _MSCanConst(_TimingConst):
    pass


class _AT91Const(_TimingConst):
    brp_min = 2
    brp_max = 128


class _FlexCanConst(_TimingConst):
    brp_max = 256


class TimingConstants(object):
    """
    Constants used to calculate the timing registers
    """

    class MCP251x8(_MCP251xConst):
        fsys = 8000000

    class MCP251x16(_MCP251xConst):
        pass

    class MPC251x32(_MCP251xConst):
        fsys = 32000000

    class MSCan32(_MSCanConst):
        fsys = 32000000

    class MSCan33(_MSCanConst):
        fsys = 33000000

    class MSCan333(_MSCanConst):
        fsys = 33300000

    class MSCan33333333(_MSCanConst):
        fsys = 33333333

    class MSCanMPC51211(_MSCanConst):
        fsys = 66660000

    class MSCanMPC51212(_MSCanConst):
        fsys = 66666666

    class AT91Ronetix(_AT91Const):
        fsys = 99532800

    class AT91100(_AT91Const):
        fsys = 100000000

    class FlexCanMX28(_FlexCanConst):
        fsys = 24000000

    class FlexCanMX6(_FlexCanConst):
        fsys = 30000000

    class FlexCan49(_FlexCanConst):
        fsys = 49875000

    class FlexCan66(_FlexCanConst):
        fsys = 66000000

    class FlexCan665(_FlexCanConst):
        fsys = 66500000

    class FlexCan666(_FlexCanConst):
        fsys = 66666666

    class FlexCanVYBRID(_FlexCanConst):
        fsys = 83368421

    class SJA1000(_TimingConst):
        tseg1_min = 1
        tseg2_min = 1

    class TIHecc(_TimingConst):
        tseg1_min = 1
        tseg2_min = 1
        brp_max = 256
        fsys = 26000000

    class RCARCan(_TimingConst):
        brp_max = 1024
        fsys = 130000000

    class ESP32V1(_ESP32Const):
        pass

    class ESP32V2(_ESP32Const):
        brp_min = 132
        brp_max = 256
        brp_inc = 4
        additional_timings = (_ESP32Const,)

    class ESP32V3(ESP32V2):
        pass

    class ESP32V4(ESP32V2):
        pass

    class ESP32S2(_ESP32Const):
        brp_max = 32768


def _get_cia_sample_point(bitrate):
    if bitrate > 800000:
        return 75.0
    if bitrate > 500000:
        return 80.0

    return 87.5


class Bitrate(object):
    """
    Bitrate timing register calculator.

    This class is used to calculate the values used on an assortment of CAN interfaces.
    It will also calculate the registers for interfaces not included in this module.

    These are the pre defined interfaces and the clock frequency they are defined for.

    MCP251x8 = 8000000
    MCP251x16 = 16000000
    MPC251x32 = 32000000
    MSCan32 = 32000000
    MSCan33 = 33000000
    MSCan333 = 33300000
    MSCan33333333 = 33333333
    MSCanMPC51211 = 66660000
    MSCanMPC51212 = 66666666
    AT91Ronetix = 99532800
    AT91100 = 100000000
    FlexCanMX28 = 24000000
    FlexCanMX6 = 30000000
    FlexCan49 = 49875000
    FlexCan66 = 66000000
    FlexCan665 = 66500000
    FlexCan666 = 66666666
    FlexCanVYBRID = 83368421
    SJA1000 = 16000000
    TIHecc = 26000000
    RCARCan = 130000000
    ESP32V1 = 80000000
    ESP32V2 = 80000000
    ESP32V3 = 80000000
    ESP32V4 = 80000000
    ESP32S2 = 80000000

    ***NOTE***
    This calculator may not return an identical match to the baudrate (bitrate)
    the user is wanting to use. This is OK and normal, the interface not be able
    to directly support a given bitrate because of it's clock and available clock
    dividers (brp). Due to how CAN-Bus has been engineered this problem has be
    taken into considertion and the specification does allow for "wiggle room".
    """

    TimingConstants = TimingConstants

    sync_seg = 1

    def __init__(
        self,
        bitrate,
        bitrate_tolerance=0.0,
        sample_point=None,
        sample_point_tolerance=0.0,
        number_of_samples=1,
        bus_length=1.0,
        transceiver_delay=150,
    ):
        """

        Simple to use tool to calculate the bit timing registers.

        .. code-block:: python

        bt = Bitrate(500000, sample_point=87.5)
        if bt.calc_bit_timing(TimingConstants.MCP251x16Const()):
            print(
                bt.bitrate,
                ':',
                hex(bt.cnf1)[2:].upper().zfill(2),
                hex(bt.cnf2)[2:].upper().zfill(2),
                hex(bt.cnf3)[2:].upper().zfill(2)
            )
        else:
            print('calculations failed')


        :param bitrate: target bitrate
        :type bitrate: int

        :param sample_point: target sample point, if no sample point is
        provided then the program will calculate what the CiA recommended
        sample point based on the provided bitrate.
        :type sample_point: int, float

        :param number_of_samples: not used
        :type number_of_samples: int

        :param bitrate_tolerance: allowed percentage deviation from the target bitrate
        :type bitrate_tolerance: float

        :param bus_length: length of the physical canbus network in meters
        :type bus_length: float, int

        :param transceiver_delay: processing delay of the nodes on the network in nanoseconds
        :type transceiver_delay: float, int
        """

        self._nominal_bitrate = bitrate
        if sample_point is None:
            self._nominal_sample_point = _get_cia_sample_point(bitrate)
        else:
            self._nominal_sample_point = float(sample_point)

        self._bitrate_tolerance = bitrate_tolerance
        self._sample_point_tolerance = sample_point_tolerance
        self._bus_length = bus_length
        self._transceiver_delay = transceiver_delay

        self._bitrate = 0
        self._sample_point = 0.0
        self._bitrate_error = 0.0
        self._sample_point_error = 0.0
        self._brp = 0
        self._tseg1 = 0
        self._tseg2 = 0
        self._fsys = 0
        self._sjw_max = 0

        if number_of_samples not in (1, 3):
            raise ValueError("number_of_samples must be 1 or 3")

        self._number_of_samples = number_of_samples

    @property
    def is_cia_sample_point(self) -> bool:
        """
        Checks to see if the sample point and bitrate
        conform to CiA (CAN in Automation) standards

        :return: `True` or `False`
        :rtype: bool
        """
        cia_sample_point = _get_cia_sample_point(self.bitrate)
        return self.sample_point == cia_sample_point

    @micropython.native
    def get_bitrates(
        self,
        timing_const: _TimingConst
    ):

        brp_min = timing_const.brp_min
        brp_max = timing_const.brp_max
        brp_inc = timing_const.brp_inc
        fsys = timing_const.fsys
        tseg1_min = timing_const.tseg1_min
        tseg1_max = timing_const.tseg1_max
        tseg2_min = timing_const.tseg2_min
        tseg2_max = timing_const.tseg2_max
        sjw_max = timing_const.sjw_max

        nominal_bitrate = self._nominal_bitrate
        bitrate_tolerance = self._bitrate_tolerance
        nominal_sample_point = self._nominal_sample_point
        sample_point_tolerance = self._sample_point_tolerance

        results = []

        bt = 1.0 / float(nominal_bitrate)

        for brp in range(brp_min, brp_max + 1, brp_inc):
            fcan = fsys / float(brp)
            tq = 1.0 / fcan
            btq = bt / tq
            btq_rounded = int(round(btq))

            for tseg1 in range(tseg1_min,  tseg1_max + 1):
                tseg2 = btq_rounded - (tseg1 + 1)

                if tseg1 < tseg2 or tseg2 > tseg2_max or tseg2 < tseg2_min:
                    continue

                err = -(btq / btq_rounded - 1)
                err = round(err * 1e4) / 1e4

                sample_point = round((tseg1 + 1) / btq_rounded * 1e4) / 100
                bitrate = int(round(nominal_bitrate * (1 - err)))

                sp_err = (abs(sample_point - nominal_sample_point) / nominal_sample_point) * 100
                br_err = (abs(bitrate - nominal_bitrate) / nominal_bitrate) * 100

                if br_err <= bitrate_tolerance and sp_err <= sample_point_tolerance:
                    match = Bitrate(
                        bitrate=nominal_bitrate,
                        sample_point=nominal_sample_point,
                        number_of_samples=self._number_of_samples,
                        bus_length=self._bus_length,
                        transceiver_delay=self._transceiver_delay,
                    )

                    match._bitrate = bitrate
                    match._sample_point = sample_point
                    match._bitrate_error = br_err
                    match._sample_point_error = sp_err
                    match._brp = brp
                    match._tseg1 = tseg1
                    match._tseg2 = tseg2
                    match._fsys = fsys
                    match._sjw_max = sjw_max

                    results += [match]

        for t_const in timing_const.additional_timings:
            results += self.get_bitrates(
                t_const
            )

        return results

    @property
    def bus_length(self) -> int:
        """
        Physical length of the CAN-Bus network

        Work in progress

        :return: length of network cabling
        :rtype: int
        """
        return self._bus_length

    @property
    def transceiver_delay(self) -> int:
        """
        Processing delay of a CAN-Bus node.

        Work in progress

        :return: nanosecond resolution of delay
        :rtype: int
        """
        return self._transceiver_delay

    @property
    def nominal_sample_point(self) -> float:
        """
        Target sample point supplied by the user
        when constructing an instance of this class.

        If no sample point was supplied then a CiA compliant
        sample point was generated based on the bitrate.

        :return: target sample point
        :rtype: float
        """
        return self._nominal_sample_point

    @property
    def sample_point(self) -> float:
        """
        Closest matching sample point after calculation of the bittiming registers.
        :return: best sample point match
        :rtype: float
        """
        return self._sample_point

    @property
    def sample_point_error(self) -> float:
        """
        Difference between the nominal sample point and the calculated sample point

        :return: difference represented as a 0-100 percentage difference.
        :rtype: float
        """
        return self._sample_point_error

    @property
    def nominal_bitrate(self) -> int:
        """
        Bitrate supplised by the user when constructing an instance of this class

        :return: user supplised bitrate
        :rtype: int
        """
        return self._nominal_bitrate

    @property
    def bitrate(self) -> int:
        """
        Calculated bitrate

        Closest matching bitrate to the user supplied bitrate.
        There is an error window that can be supplied when constructing this class.

        :return: calculated bitrate
        :rtype: int
        """
        return self._bitrate

    @property
    def bitrate_error(self) -> float:
        """
        Difference between the nominal bitrate and the calculated bitrate

        :return: difference represented as a 0-100 percentage difference.
        :rtype: float
        """
        return self._bitrate_error

    @property
    def tq(self) -> float:
        """
        Time Quantum

        The length of the time quantum (tq), which is the basic time unit of the bit time,
        is defined by the CAN controller’s system clock fsys and the Baud Rate Prescaler (brp)

        :return: tq
        :rtype: float
        """
        if self._fsys == 0:
            return 0.0

        fcan = self._fsys / self._brp
        tq = 1.0 / fcan
        return tq

    @property
    def btq(self) -> int:
        """
        number of time quanta to send /recv a bit

        :rtype: int
        """
        bt = 1.0 / float(self._bitrate)
        btq = bt / self.tq
        btq_rounded = int(round(btq))

        return btq_rounded

    @property
    def prop_delay(self) -> int:
        """
        Total delay caused by physical devices and media.

        :return: delay
        :rtype: int
        """
        prop_delay = 2 * ((self._bus_length * 5) + self._transceiver_delay)
        return prop_delay

    @property
    def prop_seg(self) -> int:
        """
        Propigation Time Segment

        Used to compensate physical delay times within the network
        :rtype: int
        """

        bus_delay = (self._bus_length * 5) * 0.000000001
        transceiver_delay = self._transceiver_delay * 0.000000001
        t_prop_seg = 2 * (transceiver_delay + bus_delay)

        prop_seg = -int(-t_prop_seg // self.tq)
        phase_seg = self.btq - prop_seg - self.sync_seg
        if phase_seg > 3:
            prop_seg += phase_seg % 2

        return prop_seg

    @property
    def phase_seg1(self) -> int:
        """
        Phase Buffer Segment 1

        Used to compensate for the oscillator tolerance.
        The phase_seg1 and phase_seg2 may be lengthened or shortened by synchronization.

        :rtype: int
        """

        bus_delay = (self._bus_length * 5) * 0.000000001
        transceiver_delay = self._transceiver_delay * 0.000000001
        t_prop_seg = 2 * (transceiver_delay + bus_delay)
        prop_seg = -int(-t_prop_seg // self.tq)
        phase_seg = self.btq - prop_seg - self.sync_seg
        if phase_seg < 3:
            return 0

        if phase_seg == 3:
            return 1

        return phase_seg // 2

    @property
    def phase_seg2(self) -> int:
        """
        Phase Buffer Segment 2

        Used to compensate for the oscillator tolerance.
        The phase_seg1 and phase_seg2 smay be lengthened or shortened by synchronization.

        :rtype: int
        """
        bus_delay = (self._bus_length * 5) * 1e-9
        transceiver_delay = self._transceiver_delay * 1e-9
        t_prop_seg = 2 * (transceiver_delay + bus_delay)
        prop_seg = -int(-t_prop_seg // self.tq)
        phase_seg = self.btq - prop_seg - self.sync_seg
        if phase_seg < 3:
            return 0

        if phase_seg == 3:
            return 2

        return phase_seg // 2

    @property
    def tseg1(self) -> int:
        """

        :rtype: int
        """
        return self._tseg1

    @property
    def tseg2(self) -> int:
        """

        :rtype: int
        """
        return self._tseg2

    @property
    def nbt(self) -> float:
        """
        Nominal Bit Time

        :rtype: float
        """

        return self.btq * self.tq

    @property
    def brp(self) -> int:
        """
        Bitrate Prescalar

        :rtype: int
        """
        return self._brp

    @property
    def oscillator_tolerance(self) -> float:
        f1 = self.sjw / (20 * self.btq)
        f2 = min(self.phase_seg1, self.phase_seg2) / (2 * ((13 * self.btq) - self.phase_seg2))
        if f1 < f2:
            return f1 * 100

        return f2 * 100

    @property
    def sjw(self) -> int:
        """
        Synchronization Jump Width

        Used to compensate for the oscillator tolerance.

        :rtype: int
        """
        return min(self.phase_seg1, self._sjw_max)

    @property
    def rjw(self):
        return self.sjw

    @property
    def number_of_samples(self) -> int:
        """
        Not used
        :return:
        """
        return self._number_of_samples

    @property
    def fsys(self) -> int:
        """
        CAN controller’s system clock (fsys)

        :return:
        """
        return self._fsys

    @property
    def btr0(self) -> int:
        """
        Bit Timing register used in sja1000 based interfaces

        :rtype: int
        """
        if self._brp == 0:
            return 0

        return self._brp - 1 + (self.sjw - 1) * 64

    @property
    def btr1(self) -> int:
        """
        Bit Timing register used in sja1000 based interfaces

        :rtype: int
        """
        if self._tseg1 == 0:
            return 0

        return self._tseg1 - 2 + (self._tseg2 - 1) * 16

    @property
    def can0bt(self) -> int:
        """
        Bit Timing register used in Silabs based interfaces

        :rtype: int
        """
        if self._brp == 0:
            return 0

        return self._brp + 64 * (self.sjw - 1) + self._tseg1 * 256 + self._tseg2 * 4096

    @property
    def canbr(self) -> int:
        """
        Bit Timing register used in Atmel based interfaces

        :rtype: int
        """

        if self._brp == 0:
            return 0

        br = self.phase_seg2 - 1
        br |= (self.phase_seg1 - 1) << 4
        br |= (self.prop_seg - 1) << 8
        br |= (self.sjw - 1) << 12
        br |= (self._brp - 1) << 16

        return br

    @property
    def canctrl(self) -> int:
        """
        Bit Timing register used in Microchip based interfaces

        :rtype: int
        """
        if self._brp == 0:
            return 0

        ctrl = (self._brp - 1) << 24
        ctrl |= (self.sjw - 1) << 22
        ctrl |= (self.phase_seg1 - 1) << 19
        ctrl |= (self.phase_seg2 - 1) << 16
        ctrl |= (self.prop_seg - 1) << 0

        return ctrl

    @property
    def cnf1(self):
        """
        Bit Timing register used in Microchip based interfaces

        :rtype: int
        """
        if self._brp == 0:
            return 0

        return self._brp - 1 + (self.sjw - 1) * 64

    @property
    def cnf2(self) -> int:
        """
        Bit Timing register used in Microchip based interfaces

        :rtype: int
        """
        if self._brp == 0:
            return 0

        return self.prop_seg - 2 + (self.phase_seg1 - 1) * 8 + 128

    @property
    def cnf3(self) -> int:
        """
        Bit Timing register used in Microchip based interfaces

        :rtype: int
        """
        if self._tseg2 == 0:
            return 0

        return self._tseg2 - 1

    @property
    def canbtc(self) -> int:
        """
        Bit Timing register used in Texas Instruments based interfaces

        :rtype: int
        """
        if self._brp == 0:
            return 0

        can_btc = (self.phase_seg2 - 1) & 0x7
        can_btc |= ((self.phase_seg1 + self.prop_seg - 1) & 0xF) << 3
        can_btc |= ((self.sjw - 1) & 0x3) << 8
        can_btc |= ((self._brp - 1) & 0xFF) << 16

        return can_btc

    @property
    def cxconr(self) -> int:
        """
        Bit Timing register used in Renesas based interfaces

        :rtype: int
        """
        if self._brp == 0:
            return 0

        cxconr = self._brp - 1 + (self.prop_seg - 1) * 32
        cxconr += self.phase_seg1 - 2 + (self._tseg2 - 1) * 8 + (self.sjw - 1) * 64
        return cxconr * 256

    @property
    def cibcr(self) -> int:
        """
        Bit Timing register used in Renesas based interfaces

        :rtype: int
        """

        def _bcr_tseg1(x):
            return (x & 0x0f) << 20

        def _bcr_bpr(x):
            return (x & 0x3ff) << 8

        def _bcr_sjw(x):
            return (x & 0x3) << 4

        def _bcr_tseg2(x):
            return x & 0x07

        if self._brp == 0:
            return 0

        bcr = (
            _bcr_tseg1(self.phase_seg1 + self.prop_seg - 1) |
            _bcr_bpr(self._brp - 1) |
            _bcr_sjw(self.sjw - 1) |
            _bcr_tseg2(self.phase_seg2 - 1)
        )
        return bcr << 8

    def __str__(self) -> str:
        res = [
            f"bitrate: {self.bitrate} bits/s",
            f"nominal bitrate: {self.nominal_bitrate} bits/s",
            f"bitrate error: {self.bitrate_error:.2f}%",
            f"cia compliant: {self.is_cia_sample_point}",
            f"sample point: {self.sample_point:.2f}%",
            f"nominal sample point: {self.nominal_sample_point:.2f}%",
            f"sample point error: {self.sample_point_error:.2f}%",
            f"number of samples: {self.number_of_samples}",
            f"bus length: {self.bus_length}m",
            f"transceiver delay: {self.transceiver_delay}ns",
            f"oscillator tolerance:{self.oscillator_tolerance:.3f}%",
            f"FSYS: {self.fsys / 1000000 if self.fsys else 0}mHz",
            f"SYNC_SEG: {self.sync_seg}",
            f"TQ: {self.tq * 1000000}µs",
            f"PROP_DELAY: {self.prop_delay}ns",
            f"PROP_SEG: {self.prop_seg}",
            f"PHASE_SEG1: {self.phase_seg1}",
            f"PHASE_SEG2: {self.phase_seg2}",
            f"NBT: {self.nbt * 1000000}µs",
            f"BTQ: {self.btq}",
            f"TSEG1: {self.tseg1}",
            f"TSEG2: {self.tseg2}",
            f"BRP: {self.brp}",
            f"SJW: {self.sjw}",
            f"BTR0: {self.btr0:02X}h",
            f"BTR1: {self.btr1:02X}h",
            f"CAN0BT: {self.can0bt:08X}h",
            f"CANBR: {self.canbr:08X}h",
            f"CANCTRL: {self.canctrl:08X}h",
            f"CNF1: {self.cnf1:02X}h",
            f"CNF2: {self.cnf2:02X}h",
            f"CNF3: {self.cnf3:02X}h",
            f"CANBTC: {self.canbtc:08X}h",
            f"CIBCR: {self.cibcr:08X}h",
            f"CxCONR: {self.cxconr:08X}h"
        ]

        return "\n".join(res)


if __name__ is '__main__':

    for key in sorted(list(TimingConstants.__dict__.keys())):
        if key.startswith('_'):
            continue

        tconst = TimingConstants.__dict__[key]
        print(tconst.__name__)
        print('******************************')
        print()

        brate = Bitrate(
            bitrate=100000,
            bitrate_tolerance=0.0,
            sample_point_tolerance=2.0,
            bus_length=5,
            transceiver_delay=200,
        )

        for item in brate.get_bitrates(tconst):
            print(item)
            print()

        print('******************************')
        print()
