// @ts-nocheck

export const ORACLE_FRAME_SCHEMA = 'openpave.oracle-roi.frame.v1';
export const PALM_LANDMARKS = [0, 5, 9, 13, 17];
export const LANDMARK_PARENTS = [
  9, 0, 1, 2, 3,
  0, 5, 6, 7,
  0, 9, 10, 11,
  0, 13, 14, 15,
  0, 17, 18, 19
];
export const BENDING_JOINTS = [
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20]
];

const EPSILON = 1e-8;

function finiteNumber(value) {
  return Number.isFinite(value);
}

function finitePoint2(point) {
  return point && finiteNumber(point.x) && finiteNumber(point.y);
}

function finitePoint3(point) {
  return finitePoint2(point) && finiteNumber(point.z);
}

function add3(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale3(vector, scalar) {
  return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function length3(vector) {
  return Math.hypot(vector[0], vector[1], vector[2]);
}

function normalize3(vector) {
  const length = length3(vector);
  return length > EPSILON ? scale3(vector, 1 / length) : null;
}

function point3(point) {
  return [point.x, point.y, point.z];
}

function mean3(points) {
  const sum = points.reduce((total, point) => add3(total, point), [0, 0, 0]);
  return scale3(sum, 1 / points.length);
}

// Jacobi diagonalization for a symmetric 3x3 covariance matrix. The returned
// vector is the eigenvector of the smallest eigenvalue: the fitted plane normal.
function smallestEigenvectorSymmetric3(matrix) {
  const a = matrix.map((row) => row.slice());
  const vectors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ];

  for (let iteration = 0; iteration < 16; iteration++) {
    const candidates = [
      [0, 1, Math.abs(a[0][1])],
      [0, 2, Math.abs(a[0][2])],
      [1, 2, Math.abs(a[1][2])]
    ];
    candidates.sort((left, right) => right[2] - left[2]);
    const [p, q, magnitude] = candidates[0];
    if (magnitude < 1e-12) break;

    const angle = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);

    for (let row = 0; row < 3; row++) {
      if (row === p || row === q) continue;
      const arp = a[row][p];
      const arq = a[row][q];
      a[row][p] = a[p][row] = cosine * arp - sine * arq;
      a[row][q] = a[q][row] = sine * arp + cosine * arq;
    }

    const app = a[p][p];
    const aqq = a[q][q];
    const apq = a[p][q];
    a[p][p] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq;
    a[q][q] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq;
    a[p][q] = a[q][p] = 0;

    for (let row = 0; row < 3; row++) {
      const vrp = vectors[row][p];
      const vrq = vectors[row][q];
      vectors[row][p] = cosine * vrp - sine * vrq;
      vectors[row][q] = sine * vrp + cosine * vrq;
    }
  }

  let minimumIndex = 0;
  if (a[1][1] < a[minimumIndex][minimumIndex]) minimumIndex = 1;
  if (a[2][2] < a[minimumIndex][minimumIndex]) minimumIndex = 2;
  return normalize3([
    vectors[0][minimumIndex],
    vectors[1][minimumIndex],
    vectors[2][minimumIndex]
  ]);
}

export function computeOracleRoi(normalizedLandmarks, padding = 1.25) {
  if (!Array.isArray(normalizedLandmarks) || normalizedLandmarks.length !== 21) {
    return { valid: false, reason: 'expected_21_normalized_landmarks' };
  }
  if (!normalizedLandmarks.every(finitePoint2)) {
    return { valid: false, reason: 'non_finite_normalized_landmark' };
  }

  const wrist = normalizedLandmarks[0];
  const middleMcp = normalizedLandmarks[9];
  const longitudinal = [middleMcp.x - wrist.x, middleMcp.y - wrist.y];
  const longitudinalLength = Math.hypot(longitudinal[0], longitudinal[1]);
  if (longitudinalLength < EPSILON) {
    return { valid: false, reason: 'degenerate_wrist_middle_axis' };
  }

  // ROI +v points from the MCPs toward the wrist; fingers therefore face up.
  const yAxis = [
    -longitudinal[0] / longitudinalLength,
    -longitudinal[1] / longitudinalLength
  ];
  const xAxis = [yAxis[1], -yAxis[0]];
  const projected = normalizedLandmarks.map((point) => ({
    x: point.x * xAxis[0] + point.y * xAxis[1],
    y: point.x * yAxis[0] + point.y * yAxis[1]
  }));
  const minX = Math.min(...projected.map((point) => point.x));
  const maxX = Math.max(...projected.map((point) => point.x));
  const minY = Math.min(...projected.map((point) => point.y));
  const maxY = Math.max(...projected.map((point) => point.y));
  const size = Math.max(maxX - minX, maxY - minY) * padding;
  if (!finiteNumber(size) || size < EPSILON) {
    return { valid: false, reason: 'degenerate_roi_extent' };
  }

  const centreProjected = [(minX + maxX) / 2, (minY + maxY) / 2];
  const center = [
    centreProjected[0] * xAxis[0] + centreProjected[1] * yAxis[0],
    centreProjected[0] * xAxis[1] + centreProjected[1] * yAxis[1]
  ];
  const sourceToRoi = [
    [xAxis[0] / size, xAxis[1] / size, 0.5 - centreProjected[0] / size],
    [yAxis[0] / size, yAxis[1] / size, 0.5 - centreProjected[1] / size]
  ];
  const roiToSource = [
    [size * xAxis[0], size * yAxis[0], center[0] - 0.5 * size * (xAxis[0] + yAxis[0])],
    [size * xAxis[1], size * yAxis[1], center[1] - 0.5 * size * (xAxis[1] + yAxis[1])]
  ];

  return {
    valid: true,
    padding,
    center,
    size,
    rotationRadians: Math.atan2(yAxis[1], yAxis[0]),
    xAxis,
    yAxis,
    sourceToRoi,
    roiToSource,
    sourceSpace: 'normalized_image_unmirrored',
    roiSpace: 'unit_square_fingers_up'
  };
}

export function computePalmFrame(worldLandmarks, maximumResidual = 0.15) {
  if (!Array.isArray(worldLandmarks) || worldLandmarks.length !== 21) {
    return { valid: false, reason: 'expected_21_world_landmarks' };
  }
  if (!PALM_LANDMARKS.every((index) => finitePoint3(worldLandmarks[index]))) {
    return { valid: false, reason: 'non_finite_palm_anchor' };
  }

  const anchors = PALM_LANDMARKS.map((index) => point3(worldLandmarks[index]));
  const origin = anchors[0];
  const centroid = mean3(anchors);
  const mcpCentroid = mean3(anchors.slice(1));
  const longitudinal = normalize3(sub3(mcpCentroid, origin));
  const lateral = normalize3(sub3(point3(worldLandmarks[5]), point3(worldLandmarks[17])));
  if (!longitudinal || !lateral) {
    return { valid: false, reason: 'degenerate_palm_axes' };
  }

  const centered = anchors.map((point) => sub3(point, centroid));
  const covariance = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  for (const point of centered) {
    for (let row = 0; row < 3; row++) {
      for (let column = 0; column < 3; column++) {
        covariance[row][column] += point[row] * point[column] / centered.length;
      }
    }
  }

  let normal = smallestEigenvectorSymmetric3(covariance);
  const signReference = normalize3(cross3(lateral, longitudinal));
  if (!normal || !signReference) {
    return { valid: false, reason: 'degenerate_plane_fit' };
  }
  if (dot3(normal, signReference) < 0) normal = scale3(normal, -1);

  // Re-orthogonalize in the fitted plane rather than trusting noisy raw axes.
  const xAxis = normalize3(sub3(lateral, scale3(normal, dot3(lateral, normal))));
  const yAxis = xAxis ? normalize3(cross3(normal, xAxis)) : null;
  if (!xAxis || !yAxis) {
    return { valid: false, reason: 'failed_frame_orthogonalization' };
  }
  if (dot3(yAxis, longitudinal) < 0) {
    for (let i = 0; i < 3; i++) {
      xAxis[i] *= -1;
      yAxis[i] *= -1;
    }
  }

  const palmScale = anchors
    .slice(1)
    .reduce((total, point) => total + length3(sub3(point, origin)), 0) / 4;
  const rmsPlaneResidual = Math.sqrt(
    centered.reduce((total, point) => total + dot3(point, normal) ** 2, 0) / centered.length
  );
  const normalizedPlaneResidual = palmScale > EPSILON ? rmsPlaneResidual / palmScale : Infinity;
  const valid = finiteNumber(normalizedPlaneResidual) && normalizedPlaneResidual <= maximumResidual;

  return {
    valid,
    reason: valid ? null : 'plane_residual_exceeds_threshold',
    origin,
    centroid,
    axes: { x: xAxis, y: yAxis, z: normal },
    palmScale,
    rmsPlaneResidual,
    normalizedPlaneResidual,
    maximumResidual,
    normalSignRule: 'pca_normal_dot_cross(pinky_to_index,wrist_to_mcp_centroid)>=0',
    normalSemantics: 'teacher_palm_plane_pseudo_normal'
  };
}

export function computeBoneFrames(worldLandmarks, palmFrame) {
  return LANDMARK_PARENTS.map((parentIndex, jointIndex) => {
    if (!palmFrame?.valid || !finitePoint3(worldLandmarks?.[jointIndex])) {
      return { jointIndex, parentIndex, valid: false, reason: 'missing_valid_palm_frame_or_joint' };
    }

    const tangent = jointIndex === 0
      ? palmFrame.axes.y.slice()
      : normalize3(sub3(point3(worldLandmarks[jointIndex]), point3(worldLandmarks[parentIndex])));
    const binormal = tangent ? normalize3(cross3(palmFrame.axes.z, tangent)) : null;
    const normal = tangent && binormal ? normalize3(cross3(tangent, binormal)) : null;
    if (!tangent || !binormal || !normal) {
      return { jointIndex, parentIndex, valid: false, reason: 'degenerate_bone_frame' };
    }
    return {
      jointIndex,
      parentIndex,
      valid: true,
      origin: point3(worldLandmarks[jointIndex]),
      axes: { tangent, binormal, normal },
      semantics: 'skeleton_local_frame_not_skin_surface'
    };
  });
}

export function computeBendingPlaneNormals(worldLandmarks, palmFrame, minimumSine = 0.05) {
  return BENDING_JOINTS.map(([jointIndex, childIndex]) => {
    const parentIndex = LANDMARK_PARENTS[jointIndex];
    if (!palmFrame?.valid || ![parentIndex, jointIndex, childIndex].every((index) => finitePoint3(worldLandmarks?.[index]))) {
      return { jointIndex, parentIndex, childIndex, valid: false, reason: 'missing_valid_geometry' };
    }
    const incoming = normalize3(sub3(point3(worldLandmarks[jointIndex]), point3(worldLandmarks[parentIndex])));
    const outgoing = normalize3(sub3(point3(worldLandmarks[childIndex]), point3(worldLandmarks[jointIndex])));
    let normal = incoming && outgoing ? normalize3(cross3(incoming, outgoing)) : null;
    const sine = incoming && outgoing ? length3(cross3(incoming, outgoing)) : 0;
    if (!normal || sine < minimumSine) {
      return { jointIndex, parentIndex, childIndex, valid: false, reason: 'near_collinear_bones', sine };
    }
    if (dot3(normal, palmFrame.axes.z) < 0) normal = scale3(normal, -1);
    return {
      jointIndex,
      parentIndex,
      childIndex,
      valid: true,
      normal,
      sine,
      signRule: 'dot(bending_normal,palm_normal)>=0',
      semantics: 'joint_bending_plane_pseudo_normal'
    };
  });
}

export function buildOracleHand(normalizedLandmarks, worldLandmarks, handednessCategories, handIndex) {
  const roi = computeOracleRoi(normalizedLandmarks);
  const palmFrame = computePalmFrame(worldLandmarks);
  const boneFrames = computeBoneFrames(worldLandmarks, palmFrame);
  const bendingPlaneNormals = computeBendingPlaneNormals(worldLandmarks, palmFrame);
  const handedness = handednessCategories?.[0] ?? null;
  const normalizedValid = Array.isArray(normalizedLandmarks) &&
    normalizedLandmarks.length === 21 && normalizedLandmarks.every(finitePoint3);
  const worldValid = Array.isArray(worldLandmarks) &&
    worldLandmarks.length === 21 && worldLandmarks.every(finitePoint3);

  return {
    handIndex,
    handedness,
    normalizedLandmarks,
    worldLandmarks,
    oracleRoi: roi,
    palmFrame,
    boneFrames,
    bendingPlaneNormals,
    validity: {
      normalizedLandmarks: normalizedValid,
      worldLandmarks: worldValid,
      oracleRoi: roi.valid,
      palmFrame: palmFrame.valid,
      validBoneFrames: boneFrames.filter((frame) => frame.valid).length,
      validBendingNormals: bendingPlaneNormals.filter((normal) => normal.valid).length,
      exportable: normalizedValid && worldValid && roi.valid && palmFrame.valid
    },
    uncertainty: {
      perJointAvailable: false,
      handednessScore: finiteNumber(handedness?.score) ? handedness.score : null,
      palmPlaneResidualNormalized: finiteNumber(palmFrame?.normalizedPlaneResidual)
        ? palmFrame.normalizedPlaneResidual
        : null,
      source: 'mediapipe_teacher_pseudo_label'
    }
  };
}
