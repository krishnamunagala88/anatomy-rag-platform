import unittest

import orchestrator


class OrchestratorCitationTests(unittest.TestCase):
    def test_build_prompt_includes_citation_instruction(self):
        prompt = orchestrator.build_prompt(
            "what is varicose veins",
            ["Context chunk one"],
        )
        self.assertIn("Citations:", prompt)
        self.assertIn("citation", prompt.lower())

    def test_append_citations_formats_retrieved_sources(self):
        answer = orchestrator.append_citations(
            "Varicose veins are abnormal veins.",
            [
                {
                    "chapter": "5. Cardiovascular System",
                    "subheading": "Varicose veins / lower limb",
                    "page_start": 131,
                }
            ],
        )
        self.assertIn("Citations:", answer)
        self.assertIn("5. Cardiovascular System > Varicose veins / lower limb (p. 131)", answer)


if __name__ == "__main__":
    unittest.main()
