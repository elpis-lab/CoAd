import copy
import xml.etree.ElementTree as ET
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FETCH_XML_IN  = "assets/fetch/fetch.xml"
FETCH_XML_OUT = "assets/fetch/fetch_two.xml"

# Attributes that refer to named objects and must be prefixed inside duplicated robot-specific sections
REF_ATTRS = {
    "name",     # object names everywhere (body/joint/geom/site/actuator/tendon)
    "joint",    # tendon/actuator refs
    "joint1", "joint2",  # equality refs
    "body1", "body2",    # contact excludes
    "tendon",  # actuator refs
    "site",    # (rare) refs
}

# Attributes that look like names but should NOT be prefixed (shared global assets / classes)
DO_NOT_PREFIX_ATTRS = {
    "class", "childclass",
    "mesh", "material", "texture", "file",
    "type", "group", "rgba", "pos", "quat",
    "axis", "range", "limited", "damping",
    "armature", "dyntype", "biastype", "gainprm", "biasprm",
    "ctrlrange", "forcerange", "solimp", "solref",
    "contype", "conaffinity", "specular", "shininess",
}

def prefix_attrs(elem: ET.Element, prefix: str):
    """Prefix selected attributes in elem and all descendants."""
    for e in elem.iter():
        for k, v in list(e.attrib.items()):
            if k in DO_NOT_PREFIX_ATTRS:
                continue
            if k in REF_ATTRS:
                # Only prefix non-empty strings
                if v and not v.startswith(prefix):
                    e.set(k, prefix + v)

def find_child(parent: ET.Element, tag: str):
    for c in parent:
        if c.tag == tag:
            return c
    return None

# Parse original fetch.xml
tree = ET.parse(FETCH_XML_IN)
root = tree.getroot()

# Grab key top-level sections
worldbody = find_child(root, "worldbody")
tendon    = find_child(root, "tendon")
equality  = find_child(root, "equality")
actuator  = find_child(root, "actuator")
contact   = find_child(root, "contact")

if worldbody is None:
    raise RuntimeError("No <worldbody> found in fetch.xml")

# In your fetch.xml, the robot is the top-level body under <worldbody>: <body name="root" childclass="fetch"> ...
robot_root_body = None
for c in list(worldbody):
    if c.tag == "body":
        robot_root_body = c
        break

if robot_root_body is None:
    raise RuntimeError("No top-level <body> found under <worldbody> in fetch.xml")

# Build a NEW MJCF root by copying everything except robot-specific sections we will regenerate
new_root = ET.Element(root.tag, root.attrib)

# Copy all top-level children except: worldbody/tendon/equality/actuator/contact
skip = {"worldbody", "tendon", "equality", "actuator", "contact"}
for c in list(root):
    if c.tag in skip:
        continue
    new_root.append(copy.deepcopy(c))

# Create new worldbody and carry over non-robot things like lights if any exist
new_worldbody = ET.SubElement(new_root, "worldbody")

# Copy any non-body elements from original worldbody (e.g., lights) into new worldbody
for c in list(worldbody):
    if c is robot_root_body:
        continue
    if c.tag != "body":
        new_worldbody.append(copy.deepcopy(c))

def make_robot_instance(prefix: str, parent_name: str, parent_pos: str):
    # Parent wrapper body for positioning
    wrapper = ET.SubElement(new_worldbody, "body", {"name": parent_name, "pos": parent_pos})

    # Clone robot body subtree
    robot = copy.deepcopy(robot_root_body)

    # Prefix ALL named items inside the robot subtree (bodies, joints, geoms, sites, etc.)
    prefix_attrs(robot, prefix)

    # Also prefix the top-level robot body name itself (already handled via name attr),
    # but keep childclass="fetch" intact (we don't touch childclass).
    wrapper.append(robot)

# Place two robots side-by-side
make_robot_instance(prefix="f1_", parent_name="fetch1", parent_pos="0 0 0")
make_robot_instance(prefix="f2_", parent_name="fetch2", parent_pos="0 0 0")

# Duplicate tendon/equality/actuator/contact sections per robot (because they reference joint/body names)
def add_prefixed_section(section: ET.Element | None, tag: str, prefix: str):
    if section is None:
        return
    sec = copy.deepcopy(section)
    prefix_attrs(sec, prefix)
    new_root.append(sec)

# Add per-robot global sections
add_prefixed_section(tendon,   "tendon",   "f1_")
add_prefixed_section(equality, "equality", "f1_")
add_prefixed_section(actuator, "actuator", "f1_")
add_prefixed_section(contact,  "contact",  "f1_")

add_prefixed_section(tendon,   "tendon",   "f2_")
add_prefixed_section(equality, "equality", "f2_")
add_prefixed_section(actuator, "actuator", "f2_")
add_prefixed_section(contact,  "contact",  "f2_")

# Write out
ET.ElementTree(new_root).write(FETCH_XML_OUT, encoding="utf-8", xml_declaration=True)
print("Wrote:", FETCH_XML_OUT)