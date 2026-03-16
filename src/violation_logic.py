def evaluate_violations(detections):

    violations = []
    unsafe_types = set()

    for det in detections:
        name = det["class_name"].lower()

        # Direct violation classes from model
        if name in [
            "no helmet",
            "no safety shoes",
            "no gloves",
            "no harness"
        ]:
            violations.append(f"Violation: {det['class_name']}")
            unsafe_types.add(name)

    summary = {
        "total_detections": len(detections),
        "total_violations": len(violations)
    }

    return violations, summary, unsafe_types
