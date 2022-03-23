import roof

filepath = '/Users/acarter/Documents/TRANSITS/ERS/DATA_CHALLENGE/ROOF/ROOF_data/NIRSpec/jwdata0010010_11010_0001_NRS1_uncal_updatedHDR_10INTS_ramp_STOPAFTERBIAS.fits'

data = roof.io.Data(filepath)
print(data.sci)

#data.finish_stage1(output_dir='.')