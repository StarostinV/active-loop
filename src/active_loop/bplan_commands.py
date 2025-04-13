
from bluesky_queueserver_api import BPlan


DEFAULT_TT_MIN = 0.10
DEFAULT_TT_MAX = 1.0
DEFAULT_NUM_POINTS = 64
DEFAULT_ACQUISITION_TIME = 0.1
DEFAULT_GPOS1 = -0.043
DEFAULT_GPOS2 = 0.027
DEFAULT_LPOS1 = -25
DEFAULT_LPOS2 = 25

# Define scan classes similar to push_xrr.py
class Scan:
    name: str

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self) -> BPlan:
        if self.args or self.kwargs:
            return BPlan(self.name, *self.command(*self.args, **self.kwargs))
        else:
            return BPlan(self.name, *self.command())

    def command(self, *args, **kwargs) -> tuple:
        raise NotImplementedError("Scan is an abstract class")


class UmvAttenuation(Scan):
    name = "umv"

    def command(self, atten: float = 1e6):
        return ("atten", atten)


class UmvMotor(Scan):
    name = "umv"

    def command(self, motor: str, value: float):
        return (motor, value2str(value))

class UmvX(UmvMotor):
    def command(self, x: float):
        return super().command("lx15a", x)

class Fscan(Scan):
    name = "fscan"
    
    def command(self, expr: str, integration_time: float, *args):
        #motors = args[::2]
        #var_names = args[1::2]
        return (expr, integration_time,*args )

#om tt lx15a gonil
class F4scan(Scan)
    name = "fscan"

    def command(self, integration_time, om: [float], tt: [float], lx15a: [float], gonil: [float] ):
        return super().command(f"v1={value2str(om)},v2={value2str(tt)},v3={value2str(lx15a)},v4={value2str(gonil)}", integration_time, "om" ,"v1","tt","v2","lx15a","v3","gonil","v4")

class AttenUse(Scan):
    name = "attenuse"

    def command(self, on: bool = True):
        string = "on" if on else "off"
        return (string, )
    

class A2scan(Scan):
    name = "a2scan"

    def command(self, 
                tt_max: float = DEFAULT_TT_MAX,
                num_points: int = DEFAULT_NUM_POINTS,
                acquisition_time: float = DEFAULT_ACQUISITION_TIME,
                tt_min: float = DEFAULT_TT_MIN,
                ):
        return (
            "om",
            value2str(tt_min / 2),
            value2str(tt_max / 2),
            "tt",
            value2str(tt_min),
            value2str(tt_max),
            num_points,
            value2str(acquisition_time)
        )
    

    
class CommandGroup:
    def __init__(self, *scans: Scan):
        self.scans = scans

    def __call__(self) -> tuple[BPlan, ...]:
        return tuple(scan() for scan in self.scans)
    
    def __iter__(self):
        for scan in self.scans:
            yield scan()
    
    def __add__(self, other: "CommandGroup"):
        return CommandGroup(*self.scans, *other.scans)


class MeasureFullXRR(CommandGroup):
    def __init__(self, 
                tt_max: float = DEFAULT_TT_MAX,
                num_points: int = DEFAULT_NUM_POINTS,
                acquisition_time: float = DEFAULT_ACQUISITION_TIME,
                tt_min: float = DEFAULT_TT_MIN,

                 ):
        super().__init__(
            UmvAttenuation(),
            AttenUse(on=True),
            A2scan(
                tt_max=tt_max, 
                num_points=num_points, 
                acquisition_time=acquisition_time, 
                tt_min=tt_min,
            ),
            UmvMotor("om", 0),
            UmvMotor("tt", 0),
            UmvAttenuation(),
        )


class CorrectAlignmentX:
    def __init__(self, 
                 gpos1: float = DEFAULT_GPOS1,
                 gpos2: float = DEFAULT_GPOS2,
                 lpos1: float = DEFAULT_LPOS1,
                 lpos2: float = DEFAULT_LPOS2):
        self.gpos1 = gpos1
        self.gpos2 = gpos2
        self.lpos1 = lpos1
        self.lpos2 = lpos2
    
    def __call__(self, x: float) -> Scan:
        x = (x - self.lpos1) / (self.lpos2 - self.lpos1)
        pos = self.gpos1 + (self.gpos2 - self.gpos1) * x
        return UmvMotor("gonil", str(round(pos, 3)))


class SetX:
    def __init__(self, correct_alignment: CorrectAlignmentX):
        self.correct_alignment = correct_alignment

    def __call__(self, x: float) -> CommandGroup:
        return CommandGroup(
            UmvAttenuation(1e6),
            UmvX(x),
            self.correct_alignment(x),
        )


def value2str(value: float, precision: int = 3):
    if isinstance(value, float):
        return str(round(value, precision))
    elif isinstance(value, list):
        strlist=[str(round(x, precision)) for x in value ]
        return '[' + ",".join(strlist) + ']' 
    else:
        return value
