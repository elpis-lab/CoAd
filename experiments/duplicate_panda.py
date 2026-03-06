import copy
import xml.etree.ElementTree as ET

PANDA_XML_IN  = "assets/franka_emika_panda/panda.xml"
PANDA_XML_OUT = "assets/franka_emika_panda/panda_two.xml"

# Attributes that refer to named objects and must be prefixed
REF_ATTRS = {
    "name",
    "joint",
    "joint1", "joint2",
    "body1", "body2",
    "tendon",
    "site",
}

# Attributes that must NOT be prefixed (shared assets, classes, mesh/material refs, numeric fields)
DO_NOT_PREFIX_ATTRS = {
    "class", "childclass",
    "mesh", "material", "texture", "file", "meshdir",
    "type", "group", "rgba", "pos", "quat", "axis",
    "range", "limited", "damping", "armature",
    "dyntype", "biastype", "gainprm", "biasprm",
    "ctrlrange", "forcerange", "solimp", "solref",
    "contype", "conaffinity", "specular", "shininess",
    "integrator", "angle", "autolimits",
}

def prefix_attrs(elem: ET.Element, prefix: str):
    """Prefix selected attributes in elem and all descendants."""
    for e in elem.iter():
        for k, v in list(e.attrib.items()):
            if k in DO_NOT_PREFIX_ATTRS:
                continue
            if k in REF_ATTRS and v:
                if not v.startswith(prefix):
                    e.set(k, prefix + v)

def find_child(parent: ET.Element, tag: str):
    for c in parent:
        if c.tag == tag:
            return c
    return None

tree = ET.parse(PANDA_XML_IN)
root = tree.getroot()

worldbody = find_child(root, "worldbody")
tendon    = find_child(root, "tendon")
equality  = find_child(root, "equality")
actuator  = find_child(root, "actuator")
contact   = find_child(root, "contact")

if worldbody is None:
    raise RuntimeError("No <worldbody> found in panda.xml")

# Panda robot root is the first <body> under <worldbody>: <body name="link0" childclass="panda"> ...
robot_root_body = None
for c in list(worldbody):
    if c.tag == "body":
        robot_root_body = c
        break
if robot_root_body is None:
    raise RuntimeError("No top-level <body> found under <worldbody> in panda.xml")

# New MJCF root: copy everything except sections we will regenerate
new_root = ET.Element(root.tag, root.attrib)

skip = {"worldbody", "tendon", "equality", "actuator", "contact"}
for c in list(root):
    if c.tag in skip:
        continue
    new_root.append(copy.deepcopy(c))

# New worldbody: keep lights and non-body items from original
new_worldbody = ET.SubElement(new_root, "worldbody")
for c in list(worldbody):
    if c is robot_root_body:
        continue
    if c.tag != "body":
        new_worldbody.append(copy.deepcopy(c))

def add_robot_instance(prefix: str, wrapper_name: str, wrapper_pos: str):
    wrapper = ET.SubElement(new_worldbody, "body", {"name": wrapper_name, "pos": wrapper_pos})
    robot = copy.deepcopy(robot_root_body)
    prefix_attrs(robot, prefix)
    wrapper.append(robot)

# Put them at the SAME place (you asked for overlap)
add_robot_instance(prefix="f1_", wrapper_name="panda1", wrapper_pos="0 0 0")
add_robot_instance(prefix="f2_", wrapper_name="panda2", wrapper_pos="0 0 0")

def add_prefixed_section(section: ET.Element | None, prefix: str):
    if section is None:
        return
    sec = copy.deepcopy(section)
    prefix_attrs(sec, prefix)
    new_root.append(sec)

# Duplicate robot-specific global sections for each instance
add_prefixed_section(tendon,   "f1_")
add_prefixed_section(equality, "f1_")
add_prefixed_section(actuator, "f1_")
add_prefixed_section(contact,  "f1_")

add_prefixed_section(tendon,   "f2_")
add_prefixed_section(equality, "f2_")
add_prefixed_section(actuator, "f2_")
add_prefixed_section(contact,  "f2_")

ET.ElementTree(new_root).write(PANDA_XML_OUT, encoding="utf-8", xml_declaration=True)
print("Wrote:", PANDA_XML_OUT)