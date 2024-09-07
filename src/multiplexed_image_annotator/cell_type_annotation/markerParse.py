import numpy as np


class MarkerParser():
    def __init__(self, strict=True):
        self.panels = {}
        self.indices = {}
        self.panels['immune_base'] = ['CD45', 'CD20', 'CD4', 'CD8', 'DAPI', 'CD11c', 'CD3']

        self.panels['immune_extended'] = ['DAPI', 'CD3', 'CD4', 'CD8', 'CD11c', 'CD20', 'CD45', 'CD68', 'CD163', 'CD56']

        self.panels['immune_full'] = ['DAPI', 'CD3', 'CD4', 'CD8', 'CD11c', 'CD15', 'CD20', 'CD45', 
                                    'CD56', 'CD68', 'CD138', 'CD163', 'FoxP3', 'GranzymeB', 'Trypase']
        
        self.panels['structure'] = ['DAPI', 'aSMA', 'CD31', 'PanCK', 'Vimentin', 'Ki67', 'CD45']

        self.panels["nerve_cell"] = ['DAPI', 'CD45', 'GFAP']

        self.immune_base = False
        self.immune_extended = False
        self.immune_full = False
        self.struct = False
        self.nerve = False

        self.strict = strict
        self.markers = []

    def _matching(self, marker_list, panel):
        matched = []
        missing = []
        for marker in panel:
            if marker in marker_list:
                # find the index of the marker in the marker_list
                matched.append(marker_list.index(marker))
            else:
                if not self.strict and len(panel) > 4:
                    missing.append(marker)
                    if len(missing) > 1:
                        print(f"Missing markers: {missing}, ", end="")
                        return None
                    else:
                        matched.append(-1)
                else:
                    print(f"Marker {marker} is not found in the list, ", end="")
                    return None

        return matched

    def parse(self, marker_file):
        # read each line of the file
        marker_list = np.loadtxt(marker_file, delimiter=',', dtype=str)

        # check replacements
        replacements = {'DNA': 'DAPI', 'CD16': 'CD15', 'CD38': 'CD138', 'CD21': 'CD20'}
        # replace the markers
        for i in range(len(marker_list)):
            if marker_list[i] in replacements:
                marker_list[i] = replacements[marker_list[i]]
        marker_list = list(marker_list)

        self.n_markers = len(marker_list)

        for panel in self.panels:
            matched = self._matching(marker_list, self.panels[panel])
            if matched:
                self.indices[panel] = matched
            else:
                print(f"{panel} panel is not applied.")
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

        if self.immune_full:
            for marker in self.panels['immune_full']:
                if marker not in self.markers:
                    self.markers.append(marker)
        if self.immune_extended:
            for marker in self.panels['immune_extended']:
                if marker not in self.markers:
                    self.markers.append(marker)
        if self.immune_base:
            for marker in self.panels['immune_base']:
                if marker not in self.markers:
                    self.markers.append(marker)
        if self.struct:
            for marker in self.panels['structure']:
                if marker not in self.markers:
                    self.markers.append(marker)
        if self.nerve:
            for marker in self.panels['nerve_cell']:
                if marker not in self.markers:
                    self.markers.append(marker)
        


# test
if __name__ == '__main__':
    marker_parser = MarkerParser()
    marker_parser.parse(r"markers.txt")
    print(marker_parser.indices)
    print(marker_parser.indices['immune_base'])
    print(marker_parser.indices['immune_extended'])
    print(marker_parser.indices['immune_full'])
    print(marker_parser.indices['structure'])
    print(marker_parser.indices['nerve_cell'])
        