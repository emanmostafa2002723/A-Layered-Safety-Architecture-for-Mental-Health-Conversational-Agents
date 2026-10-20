"""Static, allowlisted instructions; never interpolate user/component prose."""
FINAL_GUIDANCE = {
    'final_boundary_clarify_v1': 'Ask one brief clarifying question about the unclear referent or intent and offer safe general support. Uncertainty is not itself a medical request; do not add an irrelevant diagnosis/medication disclaimer.',
    'final_boundary_child_v1': 'Prioritize the child’s present safety, including when an adult reports concern. Ask whether the child is safe now; encourage a safe trusted adult or appropriate local safeguarding help. In immediate danger encourage emergency help. Do not investigate, assume guilt, recommend confrontation, or prepend a diagnosis/medication disclaimer.',
    'final_boundary_restricted_v1': 'State only the applicable medical limit, then offer safe general information. Never select a diagnosis, medicine, dose, medication change, or individualized clinical treatment.',
}
