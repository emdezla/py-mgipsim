import json
class BaseInputs:
    def toJSON(self, path):
        """ Converts input signals to JSON serializable dictionary.
        """
        export = {}
        for attribute in self.__dict__:
            export[attribute] = getattr(self, attribute).toJSON()
        with open(path, "w") as f:
            json.dump(export, f, indent=4)
        return export