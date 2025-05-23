import numpy as np


class MarkerParser():
    def __init__(self, strict=True, logger=None):
        self.panels = {}
        self.indices = {}
        self.panels['immune_base'] = ['CD45', 'CD20', 'CD4', 'CD8', 'DAPI', 'CD11c', 'CD3']

        self.panels['immune_extended'] = ['DAPI', 'CD3', 'CD4', 'CD8', 'CD11c', 'CD20', 'CD45', 'CD68', 'CD163', 'CD56']

        self.panels['immune_full'] = ['DAPI', 'CD3', 'CD4', 'CD8', 'CD11c', 'CD15', 'CD20', 'CD45', 
                                    'CD56', 'CD68', 'CD138', 'CD163', 'FoxP3', 'Granzyme B', 'Trypase']
        
        self.panels['structure'] = ['DAPI', 'aSMA', 'CD31', 'PanCK', 'Vimentin', 'Ki67', 'CD45']

        self.panels["nerve_cell"] = ['DAPI', 'CD45', 'GFAP']

        self.immune_base = False
        self.immune_extended = False
        self.immune_full = False
        self.struct = False
        self.nerve = False

        self.strict = strict
        self.markers = []

        self.logger = logger

    def _matching(self, marker_list, panel, panel_name):
        matched = []
        missing = []
        thresh = {"immune_base": 1, "immune_extended": 2, "immune_full": 3, "structure": 1, "nerve_cell": 0}
        for marker in panel:
            if marker in marker_list:
                # find the index of the marker in the marker_list
                matched.append(marker_list.index(marker))
            else:
                if marker == "CD20":
                    marker_ = "CD20 or CD79a"
                elif marker == "GFAP":
                    marker_ = "GFAP or Chromogranin A"
                elif marker == "CD138":
                    marker_ = "CD138 or CD38"
                else:
                    marker_ = marker
                if not self.strict and len(panel) > 3:
                    missing.append(marker_)
                    matched.append(-1)
                    if len(missing) > thresh[panel_name]:
                        str_missing = ', '.join(missing)
                        print(f"Markers {str_missing} are not found in the list, ", end="")
                        self.logger.log(f"Markers {str_missing} are not found in the list.")
                        return None
                else:
                    print(f"Marker {marker_} is not found in the list, ", end="")
                    self.logger.log(f"Marker {marker_} is not found in the list.")
                    return None

        return matched

    def parse(self, marker_file):
        # read each line of the file
        marker_list = np.loadtxt(marker_file, delimiter=',', dtype=str)

        text = "The panel contains the following markers: "
        for marker in marker_list:
            text += marker + ", "
            self.markers.append(marker)
        text = text[:-2] + "."
        if self.logger:
            self.logger.log(text)


        # check replacements
        replacements = {'DNA': 'DAPI', 'DPAI-02': 'DAPI', 'CD16': 'CD15', 'CD38': 'CD138', 'CD79': 'CD20', 'CHGA': 'GFAP', 'SMActin': 'aSMA',
                        'CD3e': 'CD3', 'CK': 'PanCK', 'CytoKeratin': 'PanCK', 'Cytokeratin': 'PanCK', 'Cytokeratin-19': 'PanCK', 'panCK': 'PanCK'}
        # replace the markers
        for i in range(len(marker_list)):
            if marker_list[i] in replacements and replacements[marker_list[i]] not in marker_list:
                old_marker = marker_list[i]
                marker_list[i] = replacements[marker_list[i]]
                if self.logger:
                    self.logger.log(f"Replaced the marker name {old_marker} with {marker_list[i]} to match our panel.")
        if self.logger:
            self.logger.log("")


        marker_list = list(marker_list)

        self.n_markers = len(marker_list)

        for panel in self.panels:
            matched = self._matching(marker_list, self.panels[panel], panel)
            if matched:
                self.indices[panel] = matched
                print(f"{panel} panel is applied.")
                if self.logger:
                    self.logger.log(f"{panel} panel is applied.")
                    self.logger.log(f"\n")
            else:
                print(f"{panel} panel is not applied.")
                if self.logger:
                    self.logger.log(f"{panel} panel is not applied.")
                    self.logger.log(f"\n")
                self.indices[panel] = None

        if self.indices['immune_base']:
            self.immune_base = True
        if self.indices['immune_extended']:
            self.immune_extended = True
        if self.indices['immune_full']:
            self.immune_full = True
        if self.indices['structure']:
            self.struct = True
        if self.indices['nerve_cell']:
            self.nerve = True
