import logging
import math
from pathlib import Path

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import os
from typing import Any, Tuple
from typing import Any, Tuple, Union
import h5py
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import gaussian, threshold_otsu

Image.MAX_IMAGE_PIXELS = 100000000000

class PipelineStep(ABC):
    """Base pipelines step"""

    def __init__(
        self,
        save_path: Union[None, str, Path] = None,
        precompute: bool = True,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Abstract class that helps with saving and loading precomputed results

        Args:
            save_path (Union[None, str, Path], optional): Base path to save results to.
                When set to None, the results are not saved to disk. Defaults to None.
            precompute (bool, optional): Whether to perform the precomputation necessary
                for the step. Defaults to True.
            link_path (Union[None, str, Path], optional): Path to link the output directory
                to. When None, no link is created. Only supported when save_path is not None.
                Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save the output of
                the precomputation to. If not specified it defaults to the output directory
                of the step when save_path is not None. Defaults to None.
        """
        assert (
            save_path is not None or link_path is None
        ), "link_path only supported when save_path is not None"

        name = self.__repr__()
        self.save_path = save_path
        if self.save_path is not None:
            self.output_dir = Path(self.save_path) / name
            self.output_key = "default_key"
            self._mkdir()
            if precompute_path is None:
                precompute_path = save_path

        if precompute:
            self.precompute(
                link_path=link_path,
                precompute_path=precompute_path)

    def __repr__(self) -> str:
        """Representation of a pipeline step.

        Returns:
            str: Representation of a pipeline step.
        """
        variables = ",".join(
            [f"{k}={v}" for k, v in sorted(self.__dict__.items())])
        return (
            f"{self.__class__.__name__}({variables})".replace(" ", "")
            .replace('"', "")
            .replace("'", "")
            .replace("..", "")
            .replace("/", "_")
        )

    def _mkdir(self) -> None:
        """Create path to output files"""
        assert (
            self.save_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    def _link_to_path(self, link_directory: Union[None, str, Path]) -> None:
        """Links the output directory to the given directory.

        Args:
            link_directory (Union[None, str, Path]): Directory to link to
        """
        if link_directory is None or Path(
                link_directory).parent.resolve() == Path(self.output_dir):
            logging.info("Link to self skipped")
            return
        assert (
            self.save_path is not None
        ), f"Linking only supported when saving is enabled, i.e. when save_path is passed in the constructor."
        if os.path.islink(link_directory):
            if os.path.exists(link_directory):
                logging.info("Link already exists: overwriting...")
                os.remove(link_directory)
            else:
                logging.critical(
                    "Link exists, but points nowhere. Ignoring...")
                return
        elif os.path.exists(link_directory):
            os.remove(link_directory)
        os.symlink(self.output_dir, link_directory, target_is_directory=True)

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information for this step

        Args:
            link_path (Union[None, str, Path], optional): Path to link the output to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to load/save the precomputation outputs. Defaults to None.
        """
        pass

    def process(
        self, *args: Any, output_name: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Main process function of the step and outputs the result. Try to saves the output when output_name is passed.

        Args:
            output_name (Optional[str], optional): Unique identifier of the passed datapoint. Defaults to None.

        Returns:
            Any: Result of the pipeline step
        """
        if output_name is not None and self.save_path is not None:
            return self._process_and_save(
                *args, output_name=output_name, **kwargs)
        else:
            return self._process(*args, **kwargs)

    @abstractmethod
    def _process(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract method that performs the computation of the pipeline step

        Returns:
            Any: Result of the pipeline step
        """

    def _get_outputs(self, input_file: h5py.File) -> Union[Any, Tuple]:
        """Extracts the step output from a given h5 file

        Args:
            input_file (h5py.File): File to load from

        Returns:
            Union[Any, Tuple]: Previously computed output of the step
        """
        outputs = list()
        nr_outputs = len(input_file.keys())

        # Legacy, remove at some point
        if nr_outputs == 1 and self.output_key in input_file.keys():
            return tuple([input_file[self.output_key][()]])

        for i in range(nr_outputs):
            outputs.append(input_file[f"{self.output_key}_{i}"][()])
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def _set_outputs(self, output_file: h5py.File,
                     outputs: Union[Tuple, Any]) -> None:
        """Save the step output to a given h5 file

        Args:
            output_file (h5py.File): File to write to
            outputs (Union[Tuple, Any]): Computed step output
        """
        if not isinstance(outputs, tuple):
            outputs = tuple([outputs])
        for i, output in enumerate(outputs):
            output_file.create_dataset(
                f"{self.output_key}_{i}",
                data=output,
                compression="gzip",
                compression_opts=9,
            )

    def _process_and_save(
        self, *args: Any, output_name: str, **kwargs: Any
    ) -> Any:
        """Process and save in the provided path as as .h5 file

        Args:
            output_name (str): Unique identifier of the the passed datapoint

        Raises:
            read_error (OSError): When the unable to read to self.output_dir/output_name.h5
            write_error (OSError): When the unable to write to self.output_dir/output_name.h5
        Returns:
            Any: Result of the pipeline step
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.h5"
        if output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            try:
                with h5py.File(output_path, "r") as input_file:
                    output = self._get_outputs(input_file=input_file)
            except OSError as read_error:
                print(f"\n\nCould not read from {output_path}!\n\n")
                raise read_error
        else:
            output = self._process(*args, **kwargs)
            try:
                with h5py.File(output_path, "w") as output_file:
                    self._set_outputs(output_file=output_file, outputs=output)
            except OSError as write_error:
                print(f"\n\nCould not write to {output_path}!\n\n")
                raise write_error
        return output


def get_tissue_mask(
    image: np.ndarray,
    n_thresholding_steps: int = 1,
    sigma: float = 0.0,
    min_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get binary tissue mask

    Args:
        image (np.ndarray):
            (m, n, 3) nd array of thumbnail RGB image
            or (m, n) nd array of thumbnail grayscale image
        n_thresholding_steps (int, optional): number of gaussian smoothign steps. Defaults to 1.
        sigma (float, optional): sigma of gaussian filter. Defaults to 0.0.
        min_size (int, optional): minimum size (in pixels) of contiguous tissue regions to keep. Defaults to 500.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            np int32 array
                each unique value represents a unique tissue region
            np bool array
                largest contiguous tissue region.
    """
    if len(image.shape) == 3:
        # grayscale thumbnail (inverted)
        thumbnail = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    else:
        thumbnail = image

    if len(np.unique(thumbnail)) == 1:
        return None, None

    for _ in range(n_thresholding_steps):

        # gaussian smoothing of grayscale thumbnail
        if sigma > 0.0:
            thumbnail = gaussian(
                thumbnail,
                sigma=sigma,
                out=None,
                mode="nearest",
                preserve_range=True)

        # get threshold to keep analysis region
        try:
            thresh = threshold_otsu(thumbnail[thumbnail > 0])
        except ValueError:  # all values are zero
            thresh = 0

        # replace pixels outside analysis region with upper quantile pixels
        thumbnail[thumbnail < thresh] = 0

    # convert to binary
    mask = 0 + (thumbnail > 0)

    # find connected components
    labeled, _ = ndimage.label(mask)

    # only keep
    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    if len(unique) != 0:
        discard = np.in1d(labeled, unique[counts < min_size])
        discard = discard.reshape(labeled.shape)
        labeled[discard] = 0
        # largest tissue region
        mask = labeled == unique[np.argmax(counts)]
        return labeled, mask
    else:
        return None, None


class TissueMask(PipelineStep):
    def _process_and_save(
            self,
            *args,
            output_name: str,
            **kwargs) -> np.ndarray:
        """Process and save in the provided path as a png image

        Args:
            output_name (str): Name of output file
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None during construction"
        output_path = self.output_dir / f"{output_name}.png"
        if output_path.exists():
            logging.info(
                "%s: Output of %s already exists, using it instead of recomputing",
                self.__class__.__name__,
                output_name,
            )
            try:
                with Image.open(output_path) as input_file:
                    output = np.array(input_file)
            except OSError as error:
                logging.critical("Could not open %s", output_path)
                raise error
        else:
            output = self._process(*args, **kwargs)
            # with Image.fromarray(np.uint8(output*255)) as output_image:
            with Image.fromarray(output) as output_image:
                output_image.save(output_path)
        return output

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information

        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "tissue_masks")


class GaussianTissueMask(TissueMask):
    """Helper class to extract tissue mask from images"""

    def __init__(
        self,
        n_thresholding_steps: int = 1,
        sigma: int = 20,
        min_size: int = 10,
        kernel_size: int = 20,
        dilation_steps: int = 1,
        background_gray_value: int = 228,
        downsampling_factor: int = 4,
        **kwargs,
    ) -> None:
        """
        Args:
            n_thresholding_steps (int, optional): Number of gaussian smoothing steps. Defaults to 1.
            sigma (int, optional): Sigma of gaussian filter. Defaults to 20.
            min_size (int, optional): Minimum size (in pixels) of contiguous tissue regions to keep. Defaults to 10.
            kernel_size (int, optional): Dilation kernel size. Defaults to 20.
            dilation_steps (int, optional): Number of dilation steps. Defaults to 1.
            background_gray_value (int, optional): Gray value of background pixels (usually high). Defaults to 228.
            downsampling_factor (int, optional): Downsampling factor from the input image
                                                 resolution. Defaults to 4.
        """
        self.n_thresholding_steps = n_thresholding_steps
        self.sigma = sigma
        self.min_size = min_size
        self.kernel_size = kernel_size
        self.dilation_steps = dilation_steps
        self.background_gray_value = background_gray_value
        self.downsampling_factor = downsampling_factor
        super().__init__(**kwargs)
        self.kernel = np.ones((self.kernel_size, self.kernel_size), "uint8")

    @staticmethod
    def _downsample(image: np.ndarray, downsampling_factor: int) -> np.ndarray:
        """Downsample an input image with a given downsampling factor
        Args:
            image (np.array): Input tensor
            downsampling_factor (int): Factor to downsample
        Returns:
            np.array: Output tensor
        """
        height, width = image.shape[0], image.shape[1]
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image

    @staticmethod
    def _upsample(
            image: np.ndarray,
            new_height: int,
            new_width: int) -> np.ndarray:
        """Upsample an input image to a speficied new height and width
        Args:
            image (np.array): Input tensor
            new_height (int): Target height
            new_width (int): Target width
        Returns:
            np.array: Output tensor
        """
        upsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return upsampled_image

    # type: ignore[override]
    def _process(self, image: np.ndarray) -> np.ndarray:
        """Return the superpixels of a given input image
        Args:
            image (np.array): Input image
        Returns:
            np.array: Extracted tissue mask
        """
        # Downsample image
        original_height, original_width = image.shape[0], image.shape[1]
        if self.downsampling_factor != 1:
            image = self._downsample(image, self.downsampling_factor)

        tissue_mask = np.zeros(shape=(image.shape[0], image.shape[1]))
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect tissue region
        while True:
            _, mask_ = get_tissue_mask(
                image,
                n_thresholding_steps=self.n_thresholding_steps,
                sigma=self.sigma,
                min_size=self.min_size,
            )
            if mask_ is None:
                break
            mask_ = cv2.dilate(
                mask_.astype(
                    np.uint8),
                self.kernel,
                iterations=self.dilation_steps)
            image_masked = mask_ * image_gray

            if image_masked[image_masked > 0].mean(
            ) < self.background_gray_value:
                tissue_mask[mask_ != 0] = 1
                image[mask_ != 0] = (255, 255, 255)
            else:
                break
        tissue_mask = tissue_mask.astype(np.uint8)

        tissue_mask = self._upsample(
            tissue_mask, original_height, original_width)
        return tissue_mask


class AnnotationPostProcessor(PipelineStep):
    def __init__(self, background_index: int, **kwargs: Any) -> None:
        self.background_index = background_index
        super().__init__(**kwargs)

    def mkdir(self) -> Path:
        """Create path to output files"""
        assert (
            self.save_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        return Path(self.save_path)

    def _process(  # type: ignore[override]
        self, annotation: np.ndarray, tissue_mask: np.ndarray
    ) -> np.ndarray:
        annotation = annotation.copy()
        annotation[~tissue_mask.astype(bool)] = self.background_index
        return annotation

    def _process_and_save(
            self,
            *args: Any,
            output_name: str,
            **kwargs: Any) -> Any:
        return self._process(*args, **kwargs)
