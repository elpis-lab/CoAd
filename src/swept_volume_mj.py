
def cube_swept_volume_xml(object_dims, object_configs, fixed=False, rgba=[0.8, 0.8, 0.8, 1]):
	name = "swept_volume"
	joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'
	
	object_configs = np.asarray(object_configs, dtype=np.float64)
	xdim, ydim, zdim = object_dims  
	if object_configs.ndim == 2:
		object_configs = object_configs[None, :, :]

	x_lower = object_configs[:, 0, 0]
	x_upper = object_configs[:, 0, 1]
	y_lower = object_configs[:, 1, 0]
	y_upper = object_configs[:, 1, 1]
	z = object_configs[:, 2, 0] 

	R_cyl = float(np.round(0.5 * np.sqrt(xdim**2 + ydim**2), 5))
	cx = 0.5 * (x_upper + x_lower)
	cy = 0.5 * (y_upper + y_lower)

	b1_size = np.array([ (x_upper - x_lower) + 2*R_cyl,
						(y_upper - y_lower),
						np.full_like(cx, zdim) ]).T   # (B,3)

	b2_size = np.array([ (x_upper - x_lower),
						(y_upper - y_lower) + 2*R_cyl,
						np.full_like(cx, zdim) ]).T   # (B,3)

	b_pos = np.stack([cx, cy, z], axis=1)  # (B,3)

	corners = np.stack([
		np.stack([x_lower, y_lower, z], axis=1),
		np.stack([x_lower, y_upper, z], axis=1),
		np.stack([x_upper, y_lower, z], axis=1),
		np.stack([x_upper, y_upper, z], axis=1),
	], axis=1)  # (B,4,3)

	b_pos0 = b_pos[0]              # numpy (3,)
	b1_size0 = b1_size[0].tolist()
	b2_size0 = b2_size[0].tolist()
	corners0 = corners[0]          # (4,3) world
	corners_local = corners0 - b_pos0  # (4,3) local coords

	sv_xml_string = f"""
	<body name="{name}" pos="{b_pos0[0]} {b_pos0[1]} {b_pos0[2]}">
		{joint_xml}

		<!-- boxes centered at body origin -->
		<geom name="sv_box1" type="box" pos="0 0 0"
			size="{b1_size0[0]/2} {b1_size0[1]/2} {b1_size0[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<geom name="sv_box2" type="box" pos="0 0 0"
			size="{b2_size0[0]/2} {b2_size0[1]/2} {b2_size0[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<!-- cylinders at local corner offsets -->
		<geom name="sv_cyl1" type="cylinder"
			pos="{corners_local[0,0]} {corners_local[0,1]} {corners_local[0,2]}"
			size="{R_cyl} {object_dims[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<geom name="sv_cyl2" type="cylinder"
			pos="{corners_local[1,0]} {corners_local[1,1]} {corners_local[1,2]}"
			size="{R_cyl} {object_dims[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<geom name="sv_cyl3" type="cylinder"
			pos="{corners_local[2,0]} {corners_local[2,1]} {corners_local[2,2]}"
			size="{R_cyl} {object_dims[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<geom name="sv_cyl4" type="cylinder"
			pos="{corners_local[3,0]} {corners_local[3,1]} {corners_local[3,2]}"
			size="{R_cyl} {object_dims[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
	</body>
	"""
	return sv_xml_string, b_pos0

    def move_swept_volume():
        ...
		